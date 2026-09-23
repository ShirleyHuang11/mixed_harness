import json

import pytest

from auto_jev import triage_store
from auto_jev.storage import RunStore
from auto_jev.triage import DEFAULT_TRIAGE_QUESTIONS, TARGET_OF
from tests.support import ScriptedJev


def summary(routed_id, task_id, state, probs, final_route, score=0.5, version=0):
    return {'routed_id': routed_id, 'task_id': task_id, 'run_id': 'r-' + routed_id, 'best_score': score,
            'route': max(probs, key=probs.get), 'route_source': 'triage', 'final_route': final_route,
            'questions_version': version,
            'decision': {'route': max(probs, key=probs.get), 'probabilities': probs, 'confidence': max(probs.values()),
                         'evidence': {'declared_action_kind': 'labels'}, 'questions': DEFAULT_TRIAGE_QUESTIONS,
                         'state': state, 'response': {}},
            'task': {'name': task_id}}


class ReplayJev(ScriptedJev):
    """Answers the route question by looking for a marker word in the criteria text."""
    def __init__(self, marker_to_route):
        super().__init__()
        self.marker_to_route = marker_to_route

    def judge(self, state, questions, model=None):
        self.requests.append({'state': state, 'questions': questions})
        text = json.dumps(questions['route']['criteria'])
        route = next((r for marker, r in self.marker_to_route.items() if marker in text and state.get('marker') == marker), 'settled')
        probs = {r: 0.0 for r in ('settled', 'open_ended', 'hybrid')}; probs[route] = 1.0
        return {'answers': {'route': {'type': 'choice', 'choice': route, 'probabilities': probs, 'confidence': 1.0},
                            'code_computes_facts': {'type': 'noul', 'noul': 0.5}, 'needs_multistep': {'type': 'noul', 'noul': 0.5}},
                'model': 'replay', 'usage': {}, 'cache_hit': False, 'elapsed_ms': 1.0, 'client_metadata': {}}


def test_default_questions_when_nothing_stored(tmp_path):
    questions, version = triage_store.current_questions(RunStore(tmp_path))
    assert questions == DEFAULT_TRIAGE_QUESTIONS and version == 0


def test_record_maps_final_route_to_target(tmp_path):
    store = RunStore(tmp_path)
    rec = triage_store.record(store, summary('a', 'ta', {'marker': 'alpha'}, {'settled': 0.9, 'open_ended': 0.1, 'hybrid': 0.0}, 'llm'))
    assert rec['target'] == TARGET_OF['llm'] == 'open_ended' and rec['triage_route'] == 'settled'
    assert triage_store.history(store) == [rec]


def test_improve_skips_below_min_history(tmp_path):
    store = RunStore(tmp_path)
    triage_store.record(store, summary('a', 'ta', {'marker': 'alpha'}, {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0}, 'jev'))
    result = triage_store.improve(store, ScriptedJev(), lambda prompt: '{}', min_history=2)
    assert result['status'] == 'skipped' and triage_store.current_questions(store)[1] == 0


def two_records(store):
    triage_store.record(store, summary('a', 'ta', {'marker': 'alpha'}, {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0}, 'llm'))
    triage_store.record(store, summary('b', 'tb', {'marker': 'beta'}, {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0}, 'hybrid'))


def better_proposal(prompt):
    payload = json.loads(prompt)
    questions = json.loads(json.dumps(payload['current_questions']))
    questions['route']['criteria']['open_ended'] += ' alpha'
    questions['route']['criteria']['hybrid'] += ' beta'
    return json.dumps(questions)


def test_better_proposal_is_accepted_and_versioned(tmp_path):
    store = RunStore(tmp_path)
    two_records(store)
    jev = ReplayJev({'alpha': 'open_ended', 'beta': 'hybrid'})
    result = triage_store.improve(store, jev, better_proposal, min_history=2)
    assert result['status'] == 'accepted' and result['agreement_before'] == 0.0 and result['agreement_after'] == 1.0
    questions, version = triage_store.current_questions(store)
    assert version == 1 and 'alpha' in questions['route']['criteria']['open_ended']
    assert (tmp_path / 'triage' / 'questions.json').exists()
    assert len(list((tmp_path / 'triage' / 'reflections').glob('*.prompt.json'))) == 1
    assert triage_store.improvements(store)[-1]['accepted'] is True


def test_worse_or_equal_proposal_is_rejected_and_file_unchanged(tmp_path):
    store = RunStore(tmp_path)
    two_records(store)
    jev = ReplayJev({'alpha': 'open_ended', 'beta': 'hybrid'})
    same = lambda prompt: json.dumps(json.loads(prompt)['current_questions'])
    result = triage_store.improve(store, jev, same, min_history=2)
    assert result['status'] == 'rejected' and 'not better' in result['reason']
    assert triage_store.current_questions(store)[1] == 0


def test_invalid_proposal_is_rejected(tmp_path):
    store = RunStore(tmp_path)
    two_records(store)
    result = triage_store.improve(store, ScriptedJev(), lambda prompt: '{"route": {"type": "choice"}}', min_history=2)
    assert result['status'] == 'rejected'
    wrong_keys = lambda prompt: json.dumps({**json.loads(prompt)['current_questions'], 'route': {'type': 'choice', 'instructions': 'x', 'criteria': {'a': 'b', 'c': 'd', 'e': 'f'}}})
    assert triage_store.improve(store, ScriptedJev(), wrong_keys, min_history=2)['status'] == 'rejected'


def test_proposer_error_is_recorded_not_raised(tmp_path):
    store = RunStore(tmp_path)
    two_records(store)
    def broken(prompt):
        raise RuntimeError('provider down')
    result = triage_store.improve(store, ScriptedJev(), broken, min_history=2)
    assert result['status'] == 'error' and 'provider down' in result['reason']
