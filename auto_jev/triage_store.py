"""Persisted triage questions with per-run history and replay-validated improvement."""
import copy
import hashlib
import json
from pathlib import Path

from .reflection import atomic_text
from .spec import validate_questions
from .storage import append_jsonl, atomic_write_json, now_iso, read_json, read_jsonl
from .triage import DEFAULT_TRIAGE_QUESTIONS, ROUTES, TARGET_OF

INSTRUCTION = ('Improve the triage question map used to route tasks before harness authoring. '
               'Return ONLY a complete JSON question map with the same three question ids (route, code_computes_facts, needs_multistep). '
               'The route question is a choice whose criteria keys are exactly settled, open_ended and hybrid; rewrite the '
               "instructions and criteria so that, given each history record's state, the route answer matches its target. "
               'Targets are the forms the evolved harnesses actually ended up as. Do not reference task names; write general criteria.')


def _dir(store):
    path = Path(store.root) / 'triage'
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_questions(store):
    stored = read_json(_dir(store) / 'questions.json', None)
    if not stored:
        return copy.deepcopy(DEFAULT_TRIAGE_QUESTIONS), 0
    return copy.deepcopy(stored['questions']), int(stored['version'])


def history(store):
    return read_jsonl(_dir(store) / 'history.jsonl')


def improvements(store):
    return read_jsonl(_dir(store) / 'improvements.jsonl')


def record(store, summary):
    decision = summary['decision']
    entry = {'routed_id': summary['routed_id'], 'task_id': summary['task_id'],
             'task_name': (summary.get('task') or {}).get('name'), 'run_id': summary['run_id'],
             'state': copy.deepcopy(decision['state']), 'probabilities': decision['probabilities'],
             'triage_route': decision['route'], 'chosen_route': summary['route'], 'route_source': summary['route_source'],
             'final_route': summary['final_route'], 'target': TARGET_OF[summary['final_route']],
             'best_score': summary['best_score'], 'questions_version': summary.get('questions_version', 0),
             'recorded_at': now_iso()}
    append_jsonl(_dir(store) / 'history.jsonl', entry)
    return entry


def replay_agreement(questions, records, jev):
    if not records:
        return 0.0
    hits = 0
    for entry in records:
        answer = jev.judge(copy.deepcopy(entry['state']), copy.deepcopy(questions))['answers']['route']
        probabilities = answer.get('probabilities') or {}
        choice = max(probabilities, key=probabilities.get) if probabilities else answer.get('choice')
        hits += choice == entry['target']
    return hits / len(records)


def _validate_proposal(text):
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    proposed = validate_questions(json.loads(text), strict=True)
    if set(proposed) != set(DEFAULT_TRIAGE_QUESTIONS):
        raise ValueError('proposal must keep the three triage question ids')
    if proposed['route']['type'] != 'choice' or set(proposed['route']['criteria']) != set(ROUTES):
        raise ValueError('route criteria keys must be exactly settled, open_ended, hybrid')
    return proposed


def improve(store, jev, proposer, *, min_history=2):
    records = history(store)
    current, version = current_questions(store)
    if len(records) < min_history:
        result = {'status': 'skipped', 'reason': f'history has {len(records)} records; need {min_history}', 'version': version}
        append_jsonl(_dir(store) / 'improvements.jsonl', {**result, 'accepted': False, 'at': now_iso()})
        return result
    payload = {'instruction': INSTRUCTION, 'vocabulary': list(ROUTES), 'current_questions': current,
               'history': [{k: entry[k] for k in ('state', 'probabilities', 'triage_route', 'target', 'best_score')} for entry in records]}
    prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    sha = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    reflections = _dir(store) / 'reflections'
    number = len(improvements(store)) + 1
    atomic_text(reflections / f'{number:04d}-{sha}.prompt.json', prompt)
    outcome = {'version': version, 'prompt_sha256': sha, 'history_size': len(records), 'at': now_iso()}
    try:
        text = proposer(prompt).strip()
        atomic_text(reflections / f'{number:04d}-{sha}.response.txt', text)
        proposed = _validate_proposal(text)
    except Exception as exc:
        status = 'error' if not isinstance(exc, (ValueError, TypeError, KeyError)) else 'rejected'
        result = {**outcome, 'status': status, 'reason': f'{type(exc).__name__}: {exc}'[:500]}
        append_jsonl(_dir(store) / 'improvements.jsonl', {**result, 'accepted': False})
        return result
    before = replay_agreement(current, records, jev)
    after = replay_agreement(proposed, records, jev)
    accepted = after > before
    result = {**outcome, 'status': 'accepted' if accepted else 'rejected', 'agreement_before': before, 'agreement_after': after,
              'proposal_sha256': hashlib.sha256(json.dumps(proposed, sort_keys=True).encode()).hexdigest()}
    if accepted:
        version += 1
        atomic_write_json(_dir(store) / 'questions.json', {'version': version, 'questions': proposed, 'accepted_at': now_iso(),
                                                             'agreement': after, 'history_size': len(records)})
        result['version'] = version
    else:
        result['reason'] = f'replay agreement {after} is not better than {before}'
    append_jsonl(_dir(store) / 'improvements.jsonl', {**result, 'accepted': accepted})
    return result
