"""Language-model calls for llm nodes: Anthropic, OpenAI-compatible, or a mock; cache; schema-checked JSON.

`LLMClient.complete(prompt, schema=None)` returns:

    {"text": str, "value": json|str, "model": str, "usage": {"input_tokens", "output_tokens"},
     "cost_usd": float|None, "elapsed_ms": float, "cache_hit": bool, "request_hash": str}

With a schema the reply must contain one JSON document that satisfies it; `value` is the parsed document.
Without one, `value` is the text. Two error classes keep infrastructure apart from candidate quality:
`LLMError` (no key, transport failure, cache miss in cache-only mode) aborts an evolution run;
`LLMOutputError` (empty, unparsable or schema-violating reply) scores the candidate down.
"""
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .schema import SchemaError, check_schema, minimal_instance, validate as validate_json

KINDS = ('mock', 'anthropic', 'openai', 'codex')
DEFAULT_KEY_ENV = {'anthropic': 'ANTHROPIC_API_KEY', 'openai': 'OPENAI_API_KEY'}
# USD per million tokens (input, output), matched by prefix on the model id the provider echoes. Others count as unpriced.
PRICES = {'claude-sonnet-5': (2.0, 10.0), 'claude-sonnet-4-5': (3.0, 15.0), 'claude-opus-5': (15.0, 75.0), 'claude-haiku-4-5': (1.0, 5.0),
          'gpt-5.5': (5.0, 30.0)}   # gpt-5.5 at litellm's list price, so subscription calls are imputed at the same rate tau2 bills the LLM-only arm


class LLMError(RuntimeError):
    """Provider failure. Infrastructure, never a candidate score."""


class LLMOutputError(ValueError):
    """The model answered, but not in the required form. A candidate-level node error."""


def extract_json(text):
    """The first complete JSON object or array in text, as a string."""
    stripped = text.strip()
    if stripped.startswith('```'):
        lines = stripped.splitlines()
        stripped = '\n'.join(lines[1:-1] if lines[-1].strip().startswith('```') else lines[1:])
    starts = [i for i in (stripped.find('{'), stripped.find('[')) if i >= 0]
    if not starts:
        raise LLMOutputError('reply contains no JSON document')
    start = min(starts)
    closer = '}' if stripped[start] == '{' else ']'
    depth, in_string, escape = 0, False, False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in '{[':
            depth += 1
        elif char in '}]':
            depth -= 1
            if depth == 0:
                if char != closer:
                    raise LLMOutputError('unbalanced JSON document')
                return stripped[start:index + 1]
    raise LLMOutputError('unterminated JSON document')


CODEX_WORKDIR = None
CODEX_TOOL_INSTRUCTION = (
    "\n\nYou may call exactly one tool or send one message. Reply with exactly one JSON object and nothing else: "
    'either {"tool": "<tool name>", "arguments": {...}} with every required argument, or {"message": "<text to the customer>"}. '
    "Available tools (JSON Schema):\n")


def codex_exec(prompt, model, timeout=180.0):
    """One `codex exec` call on the user's ChatGPT subscription: prompt text in, the agent's last message out."""
    global CODEX_WORKDIR
    if CODEX_WORKDIR is None:
        CODEX_WORKDIR = Path(tempfile.mkdtemp(prefix='mixedharness-codex-'))
    last_error = ''
    for attempt in range(4):   # the CLI occasionally fails to start under concurrency ("failed to initialize in-process"); retry with backoff
        out = CODEX_WORKDIR / (uuid.uuid4().hex + '.txt')
        cmd = ['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '-s', 'read-only', '-C', str(CODEX_WORKDIR), '-m', model, '-o', str(out), '-']
        try:
            proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last_error = 'timed out'
            time.sleep(2 + 3 * attempt)
            continue
        text = out.read_text().strip() if out.exists() else ''
        if out.exists():
            out.unlink()
        if text:
            return text
        last_error = proc.stderr[-300:]
        time.sleep(2 + 3 * attempt)
    raise LLMError(f'codex exec returned nothing after 4 attempts: {last_error}')


class LLMClient:
    def __init__(self, kind='mock', model='mock-llm', *, api_key_env=None, cache_dir=None, cache_namespace=None,
                 cache_only=False, timeout=60.0, max_tokens=1024, mock_responder=None, base_url=None):
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}')
        self.kind, self.model, self.timeout, self.max_tokens = kind, model, timeout, max_tokens
        self.mock = kind == 'mock'
        self.mock_responder = mock_responder
        self.base_url = base_url
        self.key_env = api_key_env or DEFAULT_KEY_ENV.get(kind)
        self.cache_only = cache_only
        self.cache_namespace = cache_namespace or uuid.uuid4().hex
        if not self.cache_namespace.replace('-', '').replace('_', '').isalnum():
            raise ValueError('cache namespace must be alphanumeric')
        self.cache_dir = Path(cache_dir) / self.cache_namespace if cache_dir else None
        self.stats = {'calls': 0, 'cache_hits': 0, 'input_tokens': 0, 'output_tokens': 0, 'elapsed_ms': 0.0,
                      'cost_usd': 0.0, 'unpriced_calls': 0}
        self.actual_models = set()
        self._lock = threading.Lock()

    @property
    def metadata(self):
        return {'transport': self.kind, 'requested_model': self.model, 'api_key_env': self.key_env, 'cost_imputed': self.kind == 'codex',
                'cache_namespace': self.cache_namespace, 'cache_dir': str(self.cache_dir) if self.cache_dir else None,
                'actual_models': sorted(self.actual_models)}

    def complete(self, prompt, schema=None, *, model=None, max_tokens=None):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('prompt must be a nonempty string')
        if schema is not None:
            check_schema(schema)
        model = model or self.model
        max_tokens = max_tokens or self.max_tokens
        started = time.perf_counter()
        instruction = prompt if schema is None else (
            prompt + '\n\nRespond with a single JSON document that satisfies this JSON Schema and nothing else:\n'
            + json.dumps(schema, sort_keys=True))
        key = hashlib.sha256(json.dumps({'kind': self.kind, 'model': model, 'prompt': instruction, 'max_tokens': max_tokens},
                                        sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        cached = self._read_cache(key)
        if cached is not None:
            with self._lock:
                self.stats['cache_hits'] += 1
            text, actual_model, usage, cost = cached['text'], cached['model'], cached['usage'], cached.get('cost_usd')
            cache_hit = True
        else:
            if self.cache_only:
                raise LLMError('cache-only client missed the cache')
            text, actual_model, usage = self._call(instruction, schema, model, max_tokens)
            cost = self._price(actual_model, usage)
            cache_hit = False
            with self._lock:
                self.stats['calls'] += 1
                self.stats['input_tokens'] += int(usage.get('input_tokens') or 0)
                self.stats['output_tokens'] += int(usage.get('output_tokens') or 0)
                if cost is None:
                    self.stats['unpriced_calls'] += 1
                else:
                    self.stats['cost_usd'] += cost
                self.actual_models.add(str(actual_model))
            self._write_cache(key, {'text': text, 'model': actual_model, 'usage': usage, 'cost_usd': cost})
        try:
            value = self._parse(text, schema)
        except LLMOutputError as exc:
            raise LLMOutputError(f'{exc}; stop_reason={usage.get("stop_reason")} blocks={usage.get("content_blocks")} output_tokens={usage.get("output_tokens")}') from None
        elapsed = (time.perf_counter() - started) * 1000
        with self._lock:
            self.stats['elapsed_ms'] += elapsed
        return {'text': text, 'value': value, 'model': actual_model, 'usage': usage, 'cost_usd': cost,
                'elapsed_ms': elapsed, 'cache_hit': cache_hit, 'request_hash': key}

    # -- one native tool-calling turn (the LLM route) -------------------------------------------------------
    def turn(self, instructions, context, *, model=None, max_tokens=None):
        """One agent turn: policy + tools + conversation -> {'kind': 'tool'|'message', 'name', 'arguments', 'text', ...}.

        context = {'policy': str, 'tools': [{'name', 'description', 'parameters': json-schema}], 'conversation': [msg, ...]}
        msg = {'role': 'user'|'assistant'|'tool', 'text': str|None, 'tool': str|None, 'arguments': dict|None,
               'result': json|None, 'error': bool}
        """
        if not isinstance(context, dict) or not isinstance(context.get('conversation'), list):
            raise ValueError('agent context must be an object with a conversation list')
        model = model or self.model
        max_tokens = max_tokens or self.max_tokens
        started = time.perf_counter()
        system = (instructions.strip() + '\n\n<policy>\n' + str(context.get('policy') or '') + '\n</policy>').strip()
        tools = list(context.get('tools') or [])
        messages = self._messages(context['conversation'])
        if self.mock:
            action, actual_model, usage = self._mock_turn(context, tools, messages), 'mock-llm', {'input_tokens': len(system) // 4, 'output_tokens': 8}
            cost = 0.0
        else:
            action, actual_model, usage = self._turn_call(system, messages, tools, model, max_tokens)
            cost = self._price(actual_model, usage)
            with self._lock:
                self.stats['calls'] += 1
                self.stats['input_tokens'] += int(usage.get('input_tokens') or 0)
                self.stats['output_tokens'] += int(usage.get('output_tokens') or 0)
                if cost is None:
                    self.stats['unpriced_calls'] += 1
                else:
                    self.stats['cost_usd'] += cost
                self.actual_models.add(str(actual_model))
        elapsed = (time.perf_counter() - started) * 1000
        with self._lock:
            self.stats['elapsed_ms'] += elapsed
        return {**action, 'model': actual_model, 'usage': usage, 'cost_usd': cost, 'elapsed_ms': elapsed, 'cache_hit': False}

    @staticmethod
    def _messages(conversation):
        """Our conversation records -> provider messages. Tool results are rendered as user-turn text, which keeps the
        history valid for every provider without tracking provider-specific tool-use ids across turns."""
        messages = []
        for m in conversation:
            role = m.get('role')
            if role == 'user':
                messages.append({'role': 'user', 'content': m.get('text') or ''})
            elif role == 'assistant':
                if m.get('tool'):
                    messages.append({'role': 'assistant', 'content': f"[tool call] {m['tool']}({json.dumps(m.get('arguments') or {}, ensure_ascii=False)})"})
                else:
                    messages.append({'role': 'assistant', 'content': m.get('text') or ''})
            elif role == 'tool':
                result = m.get('result')
                text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                messages.append({'role': 'user', 'content': f"[tool result{' (error)' if m.get('error') else ''}] {m.get('tool')}: {text}"})
        merged = []
        for m in messages:   # providers require alternating roles; merge same-role neighbours
            if merged and merged[-1]['role'] == m['role']:
                merged[-1] = {'role': m['role'], 'content': merged[-1]['content'] + '\n\n' + m['content']}
            else:
                merged.append(dict(m))
        if not merged or merged[0]['role'] != 'user':
            merged.insert(0, {'role': 'user', 'content': '(conversation start)'})
        if merged[-1]['role'] != 'user':
            merged.append({'role': 'user', 'content': '(your turn)'})
        return merged

    def _mock_turn(self, context, tools, messages):
        if self.mock_responder is not None:
            return self.mock_responder(context, tools)
        return {'kind': 'message', 'name': None, 'arguments': None, 'text': 'mock agent reply'}

    def _turn_call(self, system, messages, tools, model, max_tokens):
        if self.kind == 'codex':
            prompt = '<system>\n' + system + '\n</system>\n' + '\n'.join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages)
            if tools:
                prompt += CODEX_TOOL_INSTRUCTION + json.dumps(tools, ensure_ascii=False)
            else:
                prompt += '\n\nReply with exactly one JSON object: {"message": "<text to the customer>"}.'
            text = codex_exec(prompt, model, timeout=max(self.timeout, 180.0))
            usage = {'input_tokens': len(prompt) // 4, 'output_tokens': len(text) // 4, 'estimated': True}
            try:
                obj = json.loads(extract_json(text))
            except (LLMOutputError, ValueError):
                obj = {'message': text}
            if isinstance(obj, dict) and obj.get('tool'):
                return {'kind': 'tool', 'name': str(obj['tool']), 'arguments': dict(obj.get('arguments') or {}), 'text': None}, 'codex:' + model, usage
            message = obj.get('message', text) if isinstance(obj, dict) else text
            return {'kind': 'message', 'name': None, 'arguments': None, 'text': str(message).strip()}, 'codex:' + model, usage
        key = os.getenv(self.key_env or '')
        if not key:
            raise LLMError(f'{self.key_env} is not set')
        try:
            if self.kind == 'anthropic':
                import anthropic
                spec_tools = [{'name': t['name'], 'description': t.get('description') or '', 'input_schema': t.get('parameters') or {'type': 'object', 'properties': {}}} for t in tools]
                client = anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=2)
                with client:
                    response = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages,
                                                      **({'tools': spec_tools} if spec_tools else {}))
                usage = {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens}
                text_parts, call = [], None
                for block in response.content:
                    if getattr(block, 'type', '') == 'tool_use' and call is None:
                        call = {'name': block.name, 'arguments': dict(block.input or {})}
                    elif getattr(block, 'type', '') == 'text':
                        text_parts.append(block.text)
                if call is not None:
                    return {'kind': 'tool', 'name': call['name'], 'arguments': call['arguments'], 'text': None}, response.model, usage
                return {'kind': 'message', 'name': None, 'arguments': None, 'text': ''.join(text_parts).strip()}, response.model, usage
            import openai
            spec_tools = [{'type': 'function', 'function': {'name': t['name'], 'description': t.get('description') or '', 'parameters': t.get('parameters') or {'type': 'object', 'properties': {}}}} for t in tools]
            client = openai.OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout, max_retries=2)
            with client:
                response = client.chat.completions.create(model=model, max_completion_tokens=max_tokens,
                                                          messages=[{'role': 'system', 'content': system}] + messages,
                                                          **({'tools': spec_tools} if spec_tools else {}))
            choice = response.choices[0].message
            usage = {'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                     'output_tokens': response.usage.completion_tokens if response.usage else 0}
            if choice.tool_calls:
                call = choice.tool_calls[0]
                try:
                    arguments = json.loads(call.function.arguments or '{}')
                except ValueError:
                    raise LLMOutputError('tool arguments were not valid JSON') from None
                return {'kind': 'tool', 'name': call.function.name, 'arguments': arguments, 'text': None}, response.model, usage
            return {'kind': 'message', 'name': None, 'arguments': None, 'text': (choice.content or '').strip()}, response.model, usage
        except (LLMError, LLMOutputError):
            raise
        except Exception as exc:
            raise LLMError(f'{self.kind} agent turn failed: {type(exc).__name__}: {str(exc)[:200]}') from None

    def _parse(self, text, schema):
        if not isinstance(text, str) or not text.strip():
            raise LLMOutputError(f'empty reply (type {type(text).__name__}, {len(text or "")} chars)')
        if schema is None:
            return text
        try:
            value = json.loads(extract_json(text))
        except ValueError as exc:
            raise LLMOutputError(f'reply is not valid JSON: {exc}; head={text[:160]!r} tail={text[-80:]!r}') from None
        try:
            validate_json(value, schema)
        except SchemaError as exc:
            raise LLMOutputError(f'reply violates the schema: {exc}') from None
        return value

    def _call(self, instruction, schema, model, max_tokens):
        if self.mock:
            if self.mock_responder is not None:
                text = self.mock_responder(instruction, schema)
            elif schema is not None:
                text = json.dumps(minimal_instance(schema))
            else:
                text = instruction.strip().splitlines()[-1]
            return text, 'mock-llm', {'input_tokens': len(instruction) // 4, 'output_tokens': len(text) // 4}
        if self.kind == 'codex':
            text = codex_exec(instruction + '\n\nReply with the answer only; no preamble, no analysis.', model, timeout=max(self.timeout, 180.0))
            return text, 'codex:' + model, {'input_tokens': len(instruction) // 4, 'output_tokens': len(text) // 4, 'estimated': True}
        key = os.getenv(self.key_env or '')
        if not key:
            raise LLMError(f'{self.key_env} is not set')
        try:
            if self.kind == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=2)
                with client:
                    response = client.messages.create(model=model, max_tokens=max_tokens,
                                                      messages=[{'role': 'user', 'content': instruction}])
                text = ''.join(block.text for block in response.content if getattr(block, 'type', '') == 'text')
                usage = {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens,
                         'stop_reason': getattr(response, 'stop_reason', None), 'content_blocks': [getattr(b, 'type', '?') for b in response.content]}
                return text, response.model, usage
            import openai
            client = openai.OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout, max_retries=2)
            with client:
                response = client.chat.completions.create(model=model, max_completion_tokens=max_tokens,
                                                          messages=[{'role': 'user', 'content': instruction}])
            text = response.choices[0].message.content or ''
            usage = {'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                     'output_tokens': response.usage.completion_tokens if response.usage else 0}
            return text, response.model, usage
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f'{self.kind} call failed: {type(exc).__name__}: {str(exc)[:200]}') from None

    @staticmethod
    def _price(model, usage):
        if str(model).startswith('codex:'):
            name = str(model)[len('codex:'):]
            prices = PRICES.get(name) or next((v for k, v in PRICES.items() if name.startswith(k)), PRICES['gpt-5.5'])
            return (usage.get('input_tokens', 0) * prices[0] + usage.get('output_tokens', 0) * prices[1]) / 1_000_000   # imputed: the subscription bills nothing
        prices = PRICES.get(model)
        if not prices:
            for prefix, value in sorted(PRICES.items(), key=lambda kv: -len(kv[0])):
                if str(model).startswith(prefix):
                    prices = value
                    break
        if not prices:
            return 0.0 if model == 'mock-llm' else None
        return (usage.get('input_tokens', 0) * prices[0] + usage.get('output_tokens', 0) * prices[1]) / 1_000_000

    def _read_cache(self, key):
        if self.cache_dir is None:
            return None
        path = self.cache_dir / (key + '.json')
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise LLMError(f'corrupt cache entry {path.name}') from None

    def _write_cache(self, key, entry):
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_dir / (key + '.tmp')
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, self.cache_dir / (key + '.json'))
