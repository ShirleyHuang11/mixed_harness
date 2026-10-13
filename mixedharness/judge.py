"""The judge behind every `jev` node: Jev over TypeSafe's API, or any model that returns next-token log-probabilities.

A judge answers `judge(state, questions)` with the typed answers described in `jev.py` (choice / noul / score, each
with probabilities). The harness never looks past that interface, so the judge is a configuration string:

    jev                        Jev, TypeSafe's judgment model (jev-1.13.0); needs TYPESAFE_API_KEY
    jev:<model>                another Jev model id
    logprob:<model>@<base>     any OpenAI-compatible chat endpoint that returns `top_logprobs` for the first token:
                               vLLM, llama.cpp server, SGLang, LM Studio, or `scripts/judge_server.py` on a plain
                               transformers model. No key needed unless the server wants one (JUDGE_API_KEY).
    mock                       deterministic pseudo-answers for plumbing tests

The log-probability judge turns each question into a multiple-choice prompt (the state as JSON, the instructions,
the options labelled A, B, C ...), reads the probability the model puts on each label as its next token, and
reports those probabilities after an optional temperature. A small open model read this way is less sharp than
Jev; `scripts/calibrate_judge.py` fits the temperature and reports the reliability table, and the threshold
table (`scripts/fit_policy.py`) is fitted per judge, so the harness hands a weaker judge fewer turns rather than
worse turns.
"""
import concurrent.futures
import json
import math
import os
import time

import httpx

from .jev import DEFAULT_MODEL, JevClient, JevError, normalise_answers

LABELS = [chr(ord('A') + i) for i in range(26)]
FLOOR = -30.0        # log-probability assigned to a label the server did not list among its top tokens
MAX_STATE_CHARS = 60_000

SYSTEM = ("You are a careful judge inside a tool-using agent. You are shown the agent's current state as JSON and one "
          "question about it. Answer with the single label of the option you choose and nothing else.")


def parse_judge_spec(spec):
    """'jev' | 'jev:<model>' | 'logprob:<model>@<base_url>' | 'mock' -> dict(kind, model, base_url)."""
    spec = (spec or 'jev').strip()
    if spec == 'mock':
        return {'kind': 'mock', 'model': 'mock', 'base_url': None}
    if spec == 'jev' or spec.startswith('jev:'):
        return {'kind': 'jev', 'model': spec[4:] or DEFAULT_MODEL, 'base_url': None}
    if spec.startswith('logprob:'):
        rest = spec[len('logprob:'):]
        if '@' not in rest:
            raise ValueError('logprob judge needs model@base_url, e.g. logprob:Qwen/Qwen3-4B@http://127.0.0.1:8790/v1')
        model, base = rest.rsplit('@', 1)
        return {'kind': 'logprob', 'model': model, 'base_url': base.rstrip('/')}
    raise ValueError(f'unknown judge {spec!r}; use jev, jev:<model>, logprob:<model>@<base_url> or mock')


def make_judge(spec='jev', *, cache_dir=None, cache_namespace=None, temperature=1.0, timeout=120.0, workers=4):
    """A judge client from its configuration string. The result has `.judge`, `.stats`, `.metadata`, `.transport`."""
    parsed = parse_judge_spec(spec)
    if parsed['kind'] == 'mock':
        return JevClient(mock=True, cache_dir=cache_dir, cache_namespace=cache_namespace)
    if parsed['kind'] == 'jev':
        return JevClient(model=parsed['model'], cache_dir=cache_dir, cache_namespace=cache_namespace, timeout=timeout)
    return LogprobJudge(parsed['model'], parsed['base_url'], cache_dir=cache_dir, cache_namespace=cache_namespace,
                        temperature=temperature, timeout=timeout, workers=workers)


def question_prompt(state, question):
    """The multiple-choice prompt for one question; returns (text, labels in order, option ids in order)."""
    kind = question['type']
    if kind == 'choice':
        ids = list(question['criteria'])
        if len(ids) > len(LABELS):
            raise JevError(f'logprob judge supports at most {len(LABELS)} options, got {len(ids)}')
        lines = [f'{LABELS[i]}. {oid}: {question["criteria"][oid]}' for i, oid in enumerate(ids)]
        ask = 'Which option is right? Answer with its letter only.'
    elif kind == 'noul':
        ids = ['yes', 'no']
        lines = ['A. yes', 'B. no']
        ask = 'Answer A for yes or B for no.'
    else:
        ids = [str(i) for i in range(len(question['criteria']))]
        lines = [f'{LABELS[i]}. level {i}: {question["criteria"][i]}' for i in range(len(ids))]
        ask = 'Which level applies? Answer with its letter only.'
    state_text = json.dumps(state, ensure_ascii=False, indent=1, default=str)
    if len(state_text) > MAX_STATE_CHARS:
        state_text = state_text[-MAX_STATE_CHARS:]
    text = f'STATE:\n{state_text}\n\nQUESTION: {question["instructions"]}\n\nOPTIONS:\n' + '\n'.join(lines) + f'\n\n{ask}'
    return text, LABELS[:len(ids)], ids


def label_logprobs(top_logprobs, labels):
    """Map each label to the best log-probability among tokens that spell it ('A', ' A', 'A.'); FLOOR when absent."""
    out = {label: FLOOR for label in labels}
    for token, lp in top_logprobs.items():
        key = str(token).strip().rstrip('.').rstrip(')').upper()
        if key in out and lp > out[key]:
            out[key] = float(lp)
    return out


def softmax(logprobs, temperature=1.0):
    t = max(float(temperature), 1e-3)
    scaled = {k: v / t for k, v in logprobs.items()}
    top = max(scaled.values())
    weights = {k: math.exp(v - top) for k, v in scaled.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def answer_from_logprobs(question, ids, probs_by_label, labels):
    probs = {oid: probs_by_label[label] for oid, label in zip(ids, labels)}
    if question['type'] == 'choice':
        pick = max(ids, key=lambda o: probs[o])
        return {'choice': pick, 'probabilities': probs, 'confidence': probs[pick]}
    if question['type'] == 'noul':
        return {'noul': probs['yes'], 'confidence': max(probs.values())}
    score = sum(int(level) * p for level, p in probs.items())
    return {'score': score, 'probabilities': probs, 'confidence': max(probs.values())}


class LogprobJudge(JevClient):
    """One chat-completions call per question, `max_tokens=1`, `logprobs=true`; the label distribution is the answer."""

    def __init__(self, model, base_url, *, cache_dir=None, cache_namespace=None, temperature=1.0, timeout=120.0, workers=4):
        super().__init__(model=model, cache_dir=cache_dir, cache_namespace=cache_namespace, timeout=timeout)
        self.base_url = base_url
        self.temperature = float(temperature)
        self.workers = max(1, int(workers))
        self.key = os.getenv('JUDGE_API_KEY') or 'none'
        self.transport = 'logprob'
        self.raw_logprobs = {}   # request_hash -> {qid: {label: logprob}}, kept for calibration runs

    @property
    def metadata(self):
        meta = super().metadata
        meta.update(base_url=self.base_url, temperature=self.temperature)
        return meta

    def judge(self, state, questions, model=None):
        result = super().judge(state, questions, model=self.model)   # the spec's jev_model names Jev; this judge has one model
        if result.get('cache_hit') and result.get('logprobs'):
            # the cache holds the raw label log-probabilities; the temperature is a view over them, so a cached
            # answer is re-read at this judge's temperature rather than the one it was first softmaxed at
            answers = {}
            for qid, question in questions.items():
                lp = result['logprobs'].get(qid)
                if lp is None:
                    return result
                _, labels, ids = question_prompt(state, question)
                answers[qid] = answer_from_logprobs(question, ids, softmax(lp['labels'], self.temperature), labels)
            result['answers'] = normalise_answers(questions, answers)
        return result

    def _call(self, model, state, questions):
        started = time.perf_counter()
        prompts = {qid: question_prompt(state, q) for qid, q in questions.items()}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {qid: pool.submit(self._one, model, text) for qid, (text, _, _) in prompts.items()}
            raw = {qid: f.result() for qid, f in futures.items()}
        answers, logprobs, usage = {}, {}, {'input_tokens': 0, 'output_tokens': 0}
        for qid, question in questions.items():
            text, labels, ids = prompts[qid]
            top, used = raw[qid]
            lps = label_logprobs(top, labels)
            logprobs[qid] = {'labels': lps, 'ids': ids}
            answers[qid] = answer_from_logprobs(question, ids, softmax(lps, self.temperature), labels)
            usage['input_tokens'] += used.get('prompt_tokens', 0)
            usage['output_tokens'] += used.get('completion_tokens', 0)
        answers = normalise_answers(questions, answers)
        return {'answers': answers, 'model': model, 'usage': usage, 'cost_usd': 0.0, 'priced_by': 'local',
                'logprobs': logprobs, 'server_ms': round((time.perf_counter() - started) * 1000, 1)}

    def _one(self, model, text):
        body = {'model': model, 'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}],
                'max_tokens': 1, 'temperature': 0, 'logprobs': True, 'top_logprobs': 20,
                'chat_template_kwargs': {'enable_thinking': False}}
        headers = {'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'}
        response = None
        for attempt in range(4):
            try:
                response = httpx.post(self.base_url + '/chat/completions', headers=headers, json=body, timeout=self.timeout)
            except httpx.HTTPError as exc:
                if attempt == 3:
                    raise JevError(f'logprob judge request failed: {type(exc).__name__}') from None
                time.sleep(0.5 * 2 ** attempt)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(0.5 * 2 ** attempt)
                continue
            break
        if response is None or not response.is_success:
            raise JevError(f'logprob judge HTTP {response.status_code if response is not None else "?"}: {response.text[:200] if response is not None else ""}')
        try:
            data = response.json()
            content = data['choices'][0]['logprobs']['content']
            first = content[0]
            top = {item['token']: float(item['logprob']) for item in first.get('top_logprobs') or []}
            if first.get('token') is not None and first.get('logprob') is not None:
                top.setdefault(first['token'], float(first['logprob']))
        except (KeyError, IndexError, TypeError, ValueError):
            raise JevError('logprob judge reply carries no top_logprobs for the first token; start the server with logprobs enabled') from None
        return top, data.get('usage') or {}
