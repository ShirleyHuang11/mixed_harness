"""Jev, TypeSafe's judgment model, over the direct API, with a content-addressed cache and a mock.

    POST https://api.typesafe.ai/v1/systemone   Authorization: Bearer <TYPESAFE_API_KEY>
    {"model": "jev-1.13.0", "state": <json>, "questions": {id: {type, instructions, criteria?}}}

`JevClient.judge(state, questions)` returns a normalised response:

    {"answers": {id: answer}, "model": str, "usage": {...}, "cost_usd": float|None,
     "elapsed_ms": float, "cache_hit": bool, "request_hash": str}

    choice answer: {"type": "choice", "choice": option, "probabilities": {option: p}, "confidence": p|None}
    score  answer: {"type": "score", "score": float, "probabilities": {level: p}, "confidence": p|None}
    noul   answer: {"type": "noul", "noul": p, "confidence": p|None}

The client raises `JevError` (infrastructure: missing key, HTTP failure, malformed reply) and never invents an
answer. The mock answers every question deterministically from a hash of the state, so pipelines can be
exercised end-to-end without a key; mock scores are meaningless and are labelled `model: "mock"`.
"""
import copy
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

import httpx

from .spec import validate_questions

ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
DEFAULT_MODEL = 'jev-1.13.0'
LIST_PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000   # docs.typesafe.ai/models; used only when the reply carries no price
RETRY_STATUSES = (429, 500, 502, 503, 504, 529)


class JevError(RuntimeError):
    """Provider or transport failure. Infrastructure, never a candidate score."""


def request_hash(model, state, questions):
    payload = json.dumps({'model': model, 'state': state, 'questions': questions}, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _unit(value, what):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise JevError(f'{what} is not a number') from None
    if not 0.0 <= number <= 1.0:
        raise JevError(f'{what} outside [0, 1]')
    return number


def normalise_answers(questions, raw):
    """Check the reply covers every question with the right shape; return typed answers."""
    if not isinstance(raw, dict):
        raise JevError('answers must be an object')
    missing = set(questions) - set(raw)
    if missing:
        raise JevError(f'reply lacks answers for {sorted(missing)}')
    answers = {}
    for qid, question in questions.items():
        item = raw[qid]
        if not isinstance(item, dict):
            raise JevError(f'answer {qid} must be an object')
        kind = question['type']
        confidence = _unit(item['confidence'], f'{qid}.confidence') if item.get('confidence') is not None else None
        if kind == 'noul':
            value = item.get('noul', item.get('probability'))
            answers[qid] = {'type': 'noul', 'noul': _unit(value, f'{qid}.noul'), 'confidence': confidence}
        elif kind == 'choice':
            options = list(question['criteria'])
            probabilities = item.get('probabilities') or {}
            probs = {opt: _unit(probabilities.get(opt, 0.0), f'{qid}.probabilities.{opt}') for opt in options}
            choice = item.get('choice')
            if choice not in options:
                choice = max(options, key=lambda o: probs[o])
            answers[qid] = {'type': 'choice', 'choice': choice, 'probabilities': probs, 'confidence': confidence}
        else:
            levels = [str(i) for i in range(len(question['criteria']))]
            probabilities = item.get('probabilities') or {}
            probs = {lvl: _unit(probabilities.get(lvl, probabilities.get(int(lvl), 0.0)), f'{qid}.probabilities.{lvl}') for lvl in levels}
            score = item.get('score')
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = sum(int(lvl) * p for lvl, p in probs.items())
            answers[qid] = {'type': 'score', 'score': score, 'probabilities': probs, 'confidence': confidence}
    return answers


def mock_answers(questions, state):
    """Deterministic pseudo-answers from a hash of the state and question id."""
    seed = hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode('utf-8')).digest()
    answers = {}
    for qid, question in questions.items():
        digest = hashlib.sha256(seed + qid.encode('utf-8')).digest()
        unit = digest[0] / 255.0
        if question['type'] == 'noul':
            answers[qid] = {'type': 'noul', 'noul': round(unit, 3), 'confidence': 0.5}
        elif question['type'] == 'choice':
            options = list(question['criteria'])
            pick = options[digest[1] % len(options)]
            probs = {o: (0.7 if o == pick else 0.3 / max(len(options) - 1, 1)) for o in options}
            answers[qid] = {'type': 'choice', 'choice': pick, 'probabilities': probs, 'confidence': 0.5}
        else:
            levels = len(question['criteria'])
            pick = digest[1] % levels
            probs = {str(i): (0.7 if i == pick else 0.3 / max(levels - 1, 1)) for i in range(levels)}
            answers[qid] = {'type': 'score', 'score': float(pick), 'probabilities': probs, 'confidence': 0.5}
    return answers


class JevClient:
    def __init__(self, api_key=None, model=DEFAULT_MODEL, *, mock=False, cache_dir=None, cache_namespace=None,
                 cache_only=False, timeout=60.0, endpoint=ENDPOINT, mock_responder=None):
        self.model = model
        self.mock = mock
        self.mock_responder = mock_responder   # mock only: (state, questions) -> partial answers, normalised like a real reply
        self.timeout = timeout
        self.endpoint = endpoint
        self.key = api_key or os.getenv('TYPESAFE_API_KEY_PERSONAL') or os.getenv('TYPESAFE_API_KEY_personal') or os.getenv('TYPESAFE_API_KEY')   # a personal key, when set, wins over the org key
        self.cache_only = cache_only
        self.cache_namespace = cache_namespace or uuid.uuid4().hex
        if not self.cache_namespace.replace('-', '').replace('_', '').isalnum():
            raise ValueError('cache namespace must be alphanumeric')
        self.cache_dir = Path(cache_dir) / self.cache_namespace if cache_dir else None
        self.stats = {'calls': 0, 'cache_hits': 0, 'input_tokens': 0, 'output_tokens': 0, 'elapsed_ms': 0.0,
                      'cost_usd': 0.0, 'unpriced_calls': 0}
        self.actual_models = set()
        self._lock = threading.Lock()
        self.transport = 'mock' if mock else 'typesafe'

    @property
    def metadata(self):
        return {'transport': self.transport, 'requested_model': self.model, 'cache_namespace': self.cache_namespace,
                'cache_dir': str(self.cache_dir) if self.cache_dir else None, 'actual_models': sorted(self.actual_models)}

    def judge(self, state, questions, model=None):
        model = model or self.model
        questions = validate_questions(copy.deepcopy(questions))
        state = copy.deepcopy(state)
        started = time.perf_counter()
        key = request_hash(model, state, questions)
        cached = self._read_cache(key)
        if cached is not None:
            with self._lock:
                self.stats['cache_hits'] += 1
            cached.update(cache_hit=True, elapsed_ms=(time.perf_counter() - started) * 1000, request_hash=key)
            return cached
        if self.cache_only:
            raise JevError('cache-only client missed the cache')
        if self.mock:
            answers = mock_answers(questions, state)
            if self.mock_responder is not None:
                answers = normalise_answers(questions, {**answers, **(self.mock_responder(state, questions) or {})})
            result = {'answers': answers, 'model': 'mock', 'usage': {}, 'cost_usd': 0.0}
        else:
            result = self._call(model, state, questions)
        result.update(cache_hit=False, elapsed_ms=(time.perf_counter() - started) * 1000, request_hash=key)
        with self._lock:
            self.stats['calls'] += 1
            self.stats['elapsed_ms'] += result['elapsed_ms']
            usage = result.get('usage') or {}
            self.stats['input_tokens'] += int(usage.get('input_tokens') or 0)
            self.stats['output_tokens'] += int(usage.get('output_tokens') or 0)
            if result.get('cost_usd') is None:
                self.stats['unpriced_calls'] += 1
            else:
                self.stats['cost_usd'] += result['cost_usd']
            if result.get('model'):
                self.actual_models.add(str(result['model']))
        self._write_cache(key, result)
        return result

    def _call(self, model, state, questions):
        if not self.key:
            raise JevError('TYPESAFE_API_KEY is not set')
        body = {'model': model, 'state': state, 'questions': questions}
        headers = {'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'}
        response = None
        for attempt in range(4):
            try:
                response = httpx.post(self.endpoint, headers=headers, json=body, timeout=self.timeout)
            except httpx.HTTPError as exc:
                if attempt == 3:
                    raise JevError(f'Jev request failed: {type(exc).__name__}') from None
                time.sleep(0.5 * 2 ** attempt)
                continue
            if response.status_code in RETRY_STATUSES and attempt < 3:
                time.sleep(0.5 * 2 ** attempt)
                continue
            break
        if response is None or not response.is_success:
            detail = ''
            try:
                detail = response.json().get('detail', {}).get('message', '')
            except Exception:
                pass
            raise JevError(f'Jev HTTP {response.status_code if response is not None else "?"}' + (f': {detail}' if detail else ''))
        try:
            data = response.json()
        except ValueError:
            raise JevError('Jev reply is not JSON') from None
        answers = normalise_answers(questions, data.get('answers'))
        raw_usage = data.get('usage') or {}
        usage = {'input_tokens': int(raw_usage.get('inputTokens', raw_usage.get('input_tokens', 0)) or 0),
                 'output_tokens': int(raw_usage.get('outputTokens', raw_usage.get('output_tokens', 0)) or 0)}
        cost = data.get('cost')
        try:
            cost = float(cost) if cost is not None else None
        except (TypeError, ValueError):
            cost = None
        if cost is None and usage['input_tokens']:
            cost = usage['input_tokens'] * LIST_PRICE_PER_INPUT_TOKEN
            priced_by = 'list_price'
        else:
            priced_by = 'provider' if cost is not None else None
        return {'answers': answers, 'model': str(data.get('model') or model), 'usage': usage, 'cost_usd': cost,
                'priced_by': priced_by, 'rounding': data.get('rounding')}

    def _read_cache(self, key):
        if self.cache_dir is None:
            return None
        path = self.cache_dir / (key + '.json')
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise JevError(f'corrupt cache entry {path.name}') from None
        return entry.get('response')

    def _write_cache(self, key, result):
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stored = {k: v for k, v in result.items() if k not in ('cache_hit', 'elapsed_ms', 'request_hash')}
        tmp = self.cache_dir / (key + '.tmp')
        tmp.write_text(json.dumps({'response': stored}, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, self.cache_dir / (key + '.json'))
