"""LLM node transport. Same shape and cache discipline as JevClient; no tools, text in, JSON or text out."""
import copy
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

from .schema import minimal_instance, validate_json_schema
from .storage import atomic_write_json

KINDS = ('mock', 'anthropic', 'openai')
DEFAULT_KEY_ENV = {'anthropic': 'ANTHROPIC_API_KEY', 'openai': 'OPENAI_DIRECT_API_KEY'}
_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.S)


class LLMProviderError(RuntimeError):
    pass


def extract_json(text):
    """Return the JSON document inside fenced or surrounding prose; raise if none is found."""
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith(('{', '[')):
        return stripped
    start = min((index for index in (stripped.find('{'), stripped.find('[')) if index >= 0), default=-1)
    end = max(stripped.rfind('}'), stripped.rfind(']'))
    if start < 0 or end < start:
        raise LLMProviderError('LLM response contains no JSON document')
    return stripped[start:end + 1]


class LLMClient:
    def __init__(self, kind='mock', model='mock-llm', api_key_env=None, cache_dir=None, cache_namespace=None,
                 cache_only=False, timeout=60, max_tokens=1024, mock_responder=None, base_url=None):
        if kind not in KINDS:
            raise ValueError('unknown LLM provider kind')
        self.kind = kind; self.model = model; self.timeout = timeout; self.max_tokens = max_tokens
        self.mock = kind == 'mock'; self.transport = kind
        self.base_url = base_url
        self.key_env = api_key_env or DEFAULT_KEY_ENV.get(kind)
        self.mock_responder = mock_responder
        self.cache_only = cache_only
        self.cache_namespace = cache_namespace or uuid.uuid4().hex
        if not self.cache_namespace.replace('-', '').isalnum():
            raise ValueError('Invalid cache namespace')
        self.cache_dir = Path(cache_dir) / self.cache_namespace if cache_dir else None
        self.stats = {'calls': 0, 'cache_hits': 0, 'input_tokens': 0, 'output_tokens': 0,
                      'elapsed_ms': 0.0, 'cost_usd': 0.0, 'unpriced_calls': 0}
        self.metadata = {'transport': kind, 'requested_model': model, 'model_version_pinned': False,
                         'cache_namespace': self.cache_namespace, 'cache_dir': str(self.cache_dir) if self.cache_dir else None,
                         'api_key_env': self.key_env}

    def complete(self, prompt, output_schema=None, model=None, max_tokens=None):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('prompt must be a nonempty string')
        start = time.perf_counter(); model = model or self.model; max_tokens = max_tokens or self.max_tokens
        payload = {'protocol': 'auto_jev-llm-v1', 'kind': self.kind, 'model': model, 'prompt': prompt,
                   'output_schema': output_schema, 'max_tokens': max_tokens}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        path = self.cache_dir / f'{key}.json' if self.cache_dir else None
        if path and path.exists():
            try:
                result = json.loads(path.read_text())
                digest = result.pop('response_digest', None)
                actual = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
                if digest != actual or result.get('request_hash') != key:
                    raise LLMProviderError('Cache response integrity check failed')
            except (ValueError, TypeError, AttributeError):
                raise LLMProviderError('Cache response integrity check failed') from None
            result['cache_hit'] = True; result['elapsed_ms'] = (time.perf_counter() - start) * 1000
            self.stats['cache_hits'] += 1
            return copy.deepcopy(result)
        if self.cache_only:
            raise LLMProviderError('No recorded response for this input; cache-only replay cannot call the API')
        text, actual_model, usage = self._call(prompt, output_schema, model, max_tokens)
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError('Empty LLM response')
        value = text.strip()
        if output_schema is not None:
            try:
                value = json.loads(extract_json(text))
                validate_json_schema(value, output_schema)
            except ValueError as exc:
                raise LLMProviderError(f'LLM output failed schema validation: {exc}') from None
        result = {'text': text, 'value': value, 'model': actual_model, 'usage': usage, 'cache_hit': False,
                  'request_hash': key, 'elapsed_ms': (time.perf_counter() - start) * 1000, 'cost_usd': None,
                  'client_metadata': {**copy.deepcopy(self.metadata), 'requested_model': model, 'actual_model': actual_model}}
        self.stats['calls'] += 1; self.stats['elapsed_ms'] += result['elapsed_ms']
        self.stats['input_tokens'] += usage.get('input_tokens', 0); self.stats['output_tokens'] += usage.get('output_tokens', 0)
        if not self.mock:
            self.stats['unpriced_calls'] += 1
        if path:
            cached = copy.deepcopy(result)
            cached['response_digest'] = hashlib.sha256(json.dumps(cached, sort_keys=True, allow_nan=False).encode()).hexdigest()
            atomic_write_json(path, cached)
        return copy.deepcopy(result)

    def _call(self, prompt, output_schema, model, max_tokens):
        if self.mock:
            if self.mock_responder is not None:
                return self.mock_responder(prompt, output_schema), 'mock-llm', {}
            if output_schema is not None:
                return json.dumps(minimal_instance(output_schema)), 'mock-llm', {}
            return prompt.strip().splitlines()[-1], 'mock-llm', {}
        key = os.getenv(self.key_env or '')
        if not key:
            raise LLMProviderError(f'Missing {self.key_env}')
        instruction = prompt if output_schema is None else (
            prompt + '\n\nRespond with a single JSON document matching this JSON Schema and nothing else:\n'
            + json.dumps(output_schema, sort_keys=True))
        try:
            if self.kind == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=key, timeout=self.timeout)
                with client:
                    response = client.messages.create(model=model, max_tokens=max_tokens,
                                                      messages=[{'role': 'user', 'content': instruction}])
                text = ''.join(block.text for block in response.content if block.type == 'text')
                usage = {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens}
                return text, response.model, usage
            import openai
            client = openai.OpenAI(api_key=key, max_retries=0, base_url=self.base_url, timeout=self.timeout)
            with client:
                response = client.chat.completions.create(model=model, max_completion_tokens=max_tokens,
                                                          messages=[{'role': 'user', 'content': instruction}])
            text = response.choices[0].message.content or ''
            usage = ({'input_tokens': response.usage.prompt_tokens, 'output_tokens': response.usage.completion_tokens}
                     if response.usage else {})
            return text, response.model, usage
        except Exception as exc:
            raise LLMProviderError(f'LLM request failed: {type(exc).__name__}') from None
