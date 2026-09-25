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
import threading
import time
import uuid
from pathlib import Path

from .schema import SchemaError, check_schema, minimal_instance, validate as validate_json

KINDS = ('mock', 'anthropic', 'openai')
DEFAULT_KEY_ENV = {'anthropic': 'ANTHROPIC_API_KEY', 'openai': 'OPENAI_API_KEY'}
# USD per million tokens (input, output). Only models listed here are priced; others count as unpriced.
PRICES = {}


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
        return {'transport': self.kind, 'requested_model': self.model, 'api_key_env': self.key_env,
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
        value = self._parse(text, schema)
        elapsed = (time.perf_counter() - started) * 1000
        with self._lock:
            self.stats['elapsed_ms'] += elapsed
        return {'text': text, 'value': value, 'model': actual_model, 'usage': usage, 'cost_usd': cost,
                'elapsed_ms': elapsed, 'cache_hit': cache_hit, 'request_hash': key}

    def _parse(self, text, schema):
        if not isinstance(text, str) or not text.strip():
            raise LLMOutputError('empty reply')
        if schema is None:
            return text
        try:
            value = json.loads(extract_json(text))
        except ValueError as exc:
            raise LLMOutputError(f'reply is not valid JSON: {exc}') from None
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
                usage = {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens}
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
        prices = PRICES.get(model)
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
