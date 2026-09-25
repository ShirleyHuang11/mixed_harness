"""Budget, evolution loop, triage, routed run, triage improvement, freeze. Mock providers only."""
import json

import pytest

from mixedharness import sandbox as sb
from mixedharness.budget import BudgetError, check_budget, episode_usage, validate_budget
from mixedharness.evolve import FAILURE_SCORE, EvolutionAborted, evaluate_spec, run_evolution
from mixedharness.freeze import FrozenMismatch, evaluate_frozen, freeze
from mixedharness.jev import JevClient
from mixedharness.llm import LLMClient
from mixedharness.reflect import Proposer, ReflectionError, build_prompt
from mixedharness.routed import run_routed
from mixedharness.runtime import PipelineError, Runtime
from mixedharness.spec import FORMAT, spec_hash
from mixedharness.store import RunStore
from mixedharness.triage import DEFAULT_TRIAGE_QUESTIONS, TaskError, probe, route_of_spec, seed_for_route, triage, validate_task
from mixedharness import triage_store

pytestmark = pytest.mark.skipif(not sb.sandbox_available(), reason='no native sandbox on this machine')

LABELS = ['big', 'small']


def episode(i):
    return {'id': f'e{i}', 'x': i, 'label': 'big' if i >= 5 else 'small'}


TRAIN, VAL = [episode(i) for i in range(8)], [episode(i) for i in range(8, 12)]

SEED = {'format': FORMAT, 'name': 'toy', 'jev_model': 'jev-1.13.0', 'llm_model': 'mock-llm', 'nodes': [
    {'id': 'ask', 'kind': 'jev', 'state': "{'x': obs['x']}", 'questions': {'size': {'type': 'choice', 'instructions': 'big or small?', 'criteria': {'big': '>= 5', 'small': '< 5'}}}}],
    'output': "nodes['ask']['size']['choice']"}
BETTER = {'format': FORMAT, 'name': 'toy-code', 'jev_model': 'jev-1.13.0', 'llm_model': 'mock-llm', 'nodes': [
    {'id': 'rule', 'kind': 'python', 'source': "def run(obs, nodes, memory):\n    return 'big' if obs['x'] >= 5 else 'small'\n", 'inputs': []}],
    'output': "nodes['rule']"}


def evaluator(spec, episode, jev, llm=None, capture_traces=True):
    rt = Runtime(spec, jev, llm)
    try:
        d = rt.run({'x': episode['x']})
    finally:
        rt.close()
    return {'episode_id': episode['id'], 'score': 1.0 if d['output'] == episode['label'] else 0.0,
            'traces': [dict(d, observation={'x': episode['x']})]}


def test_budget_usage_and_checks():
    result = {'traces': [{'elapsed_ms': 10, 'trace': [{'kind': 'jev', 'status': 'ok', 'response': {'cost_usd': 0.002}}]},
                         {'elapsed_ms': 30, 'trace': [{'kind': 'llm', 'status': 'ok', 'response': {'cost_usd': None}}]}]}
    usage = episode_usage(result)
    assert usage == {'latency_ms_p95': 30.0, 'latency_ms_median': 10.0, 'decisions': 2, 'cost_usd': 0.002, 'unpriced_calls': 1}
    assert check_budget(usage, {'latency_ms_p95': 20}) == ['latency_ms_p95 30 > 20']
    assert check_budget(usage, {'cost_usd_per_episode': 0.01}) == []
    with pytest.raises(BudgetError):
        validate_budget({'nope': 1})


def test_evolution_accepts_a_better_whole_spec(tmp_path):
    prompts = []

    def responder(prompt):
        prompts.append(json.loads(prompt))
        return json.dumps(BETTER)
    store = RunStore(tmp_path / 'runs')
    s = run_evolution(TRAIN, VAL, jev=JevClient(mock=True), llm=LLMClient(kind='mock'), proposer=Proposer({'kind': 'mock', 'responder': responder}),
                      evaluator=evaluator, store=store, seed_spec=SEED, task={'id': 'toy', 'objective': 'label numbers'}, rounds=2, batch_size=4)
    assert s['best_hash'] == spec_hash(BETTER) and s['best_score'] == 1.0 and s['accepted'] == 1 and s['reflections'] == 2
    payload = prompts[0]
    assert payload['parent_spec']['name'] == 'toy' and len(payload['training_trajectories']) == 4
    assert payload['training_trajectories'][0]['result']['traces'][0]['trace'][0]['response']['model'] == 'mock'
    run = store.read_run(s['run_id'])
    assert run['status'] == 'completed' and set(store.candidates(s['run_id'])) == {spec_hash(SEED), spec_hash(BETTER)}
    assert len(store.evaluations(s['run_id'])) == s['total_metric_calls']
    assert (store.run_dir(s['run_id']) / 'reflections' / '001.response.txt').is_file()


def test_evolution_rejects_proposals_that_change_fixed_fields_or_are_invalid(tmp_path):
    changed = dict(BETTER, jev_model='other')
    store = RunStore(tmp_path / 'runs')
    replies = iter([json.dumps(changed), 'not json at all'])
    s = run_evolution(TRAIN, VAL, jev=JevClient(mock=True), llm=LLMClient(kind='mock'), proposer=Proposer({'kind': 'mock', 'responder': lambda p: next(replies)}),
                      evaluator=evaluator, store=store, seed_spec=SEED, task={'id': 'toy'}, rounds=2, batch_size=4)
    assert s['accepted'] == 0 and s['proposals_valid'] == 0
    reasons = [e['reason'] for e in store.events(s['run_id'], 'proposal_rejected')]
    assert any('jev_model' in r for r in reasons) and len(reasons) == 2


def test_evolution_aborts_on_infrastructure_failure(tmp_path):
    store = RunStore(tmp_path / 'runs')
    with pytest.raises(EvolutionAborted):
        run_evolution(TRAIN, VAL, jev=JevClient(mock=False, api_key=None), llm=LLMClient(kind='mock'),
                      proposer=Proposer({'kind': 'mock', 'responder': lambda p: json.dumps(BETTER)}),
                      evaluator=evaluator, store=store, seed_spec=SEED, task={'id': 'toy'}, rounds=1, batch_size=2)
    run_id = store.list_runs()[0]
    assert store.read_run(run_id)['status'] == 'aborted' and store.events(run_id, 'aborted')


def test_evolution_scores_candidate_errors_without_aborting(tmp_path):
    def flaky(spec, episode, jev, llm=None, capture_traces=True):
        if episode['x'] == 3:
            raise PipelineError('candidate bug', partial={'traces': []}, causes=[ValueError('bad')])
        return evaluator(spec, episode, jev, llm, capture_traces)
    store = RunStore(tmp_path / 'runs')
    s = run_evolution(TRAIN, VAL, jev=JevClient(mock=True), llm=LLMClient(kind='mock'), proposer=Proposer({'kind': 'mock', 'responder': lambda p: json.dumps(BETTER)}),
                      evaluator=flaky, store=store, seed_spec=SEED, task={'id': 'toy'}, rounds=1, batch_size=8)
    rows = store.evaluations(s['run_id'])
    assert any(r['status'] == 'error' and r['score'] == FAILURE_SCORE for r in rows)


def test_reflection_prompt_cap_is_enforced():
    with pytest.raises(ReflectionError):
        build_prompt({'id': 't'}, SEED, [{'big': 'x' * 5000}], max_bytes=1000)


def test_evaluate_spec_and_freeze_roundtrip(tmp_path):
    jev = JevClient(mock=True)
    scored = evaluate_spec(BETTER, VAL, jev=jev, evaluator=evaluator)
    assert scored['mean'] == 1.0 and scored['completed'] == 4
    artifact = freeze(BETTER, evaluator=evaluator, jev=jev)
    assert artifact['spec_hash'] == spec_hash(BETTER) and artifact['runtime']['available']
    assert evaluate_frozen(artifact, VAL, evaluator=evaluator, jev=jev)['mean'] == 1.0
    tampered = dict(artifact, spec=SEED)
    with pytest.raises(FrozenMismatch):
        evaluate_frozen(tampered, VAL, evaluator=evaluator, jev=jev)
    with pytest.raises(FrozenMismatch):
        evaluate_frozen(artifact, VAL, evaluator=lambda *a, **k: None, jev=jev)


# -- triage and routing -----------------------------------------------------------------------------------
TASK = {'name': 'sizes', 'objective': 'Say whether a number is big or small.', 'observation_fields': ['x'],
        'action_space': {'kind': 'labels', 'labels': LABELS}, 'evaluator': {'kind': 'exact_match'}, 'decisions_per_episode': 1,
        'latency_budget_ms': 1000, 'examples': [{'observation': {'x': 9}, 'expected': 'big'}, {'observation': {'x': 1}, 'expected': 'small'}]}


def test_probe_is_deterministic_evidence():
    evidence = probe(TASK)
    assert evidence['declared_action_kind'] == 'labels' and evidence['label_count'] == 2 and evidence['expected_kind'] == 'label'
    assert evidence['exact_validator'] and evidence['example_count'] == 2
    with pytest.raises(TaskError):
        validate_task(dict(TASK, action_space={'kind': 'labels'}))


def test_triage_returns_distribution_and_state():
    decision = triage(TASK, JevClient(mock=True))
    assert decision['route'] in ('settled', 'open_ended', 'hybrid') and abs(sum(decision['probabilities'].values()) - 1.0) < 1e-6
    assert decision['state']['evidence'] == probe(TASK) and decision['questions'] == DEFAULT_TRIAGE_QUESTIONS


@pytest.mark.parametrize('route,kinds', [('settled', 'jev'), ('open_ended', 'llm'), ('hybrid', 'hybrid')])
def test_seed_templates_run_under_mock_providers(route, kinds):
    task = dict(TASK)
    if route != 'settled':
        task = dict(task, action_space={'kind': 'open'}, output_schema={'type': 'object', 'required': ['text'], 'properties': {'text': {'type': 'string'}}})
    spec = seed_for_route(route, task)
    assert route_of_spec(spec) == kinds
    rt = Runtime(spec, JevClient(mock=True), LLMClient(kind='mock'))
    try:
        result = rt.run({'x': 7, 'labels': LABELS})
    finally:
        rt.close()
    assert result['status'] == 'ok' and result['output'] is not None


def label_evaluator(spec, episode, jev, llm=None, capture_traces=True):
    rt = Runtime(spec, jev, llm)
    try:
        d = rt.run({'x': episode['x'], 'labels': LABELS})
    finally:
        rt.close()
    out = d['output']
    return {'episode_id': episode['id'], 'score': 1.0 if out == episode['label'] else 0.0, 'traces': [dict(d, observation={'x': episode['x']})]}


def test_routed_run_records_routes_and_improves_triage_after_two_runs(tmp_path):
    better = seed_for_route('settled', TASK, llm_model='mock-llm')
    better['llm_model'] = 'mock-llm'
    better['nodes'][0]['source'] = ('def run(obs, nodes, memory):\n    rule = "big" if obs["x"] >= 5 else "small"\n'
                                    '    return {"options": obs["labels"], "questions": {"action": {"type": "choice", "instructions": "confirm", "criteria": {rule: "the rule"}}}}\n')
    better['nodes'][2]['source'] = 'def run(obs, nodes, memory):\n    return nodes["pick"]["action"]["choice"]\n'
    improved_questions = json.loads(json.dumps(DEFAULT_TRIAGE_QUESTIONS))
    improved_questions['route']['instructions'] = 'Improved instructions.'

    def responder(prompt):
        payload = json.loads(prompt)
        return json.dumps(improved_questions if 'current_questions' in payload else better)
    store = RunStore(tmp_path / 'runs')
    proposer = Proposer({'kind': 'mock', 'responder': responder})
    first = run_routed(TASK, TRAIN, VAL, jev=JevClient(mock=True), llm=LLMClient(kind='mock'), proposer=proposer, store=store,
                       evaluator=label_evaluator, task_id='sizes', rounds=1, batch_size=4)
    assert first['route_source'] == 'triage' and first['seed_route'] in ('jev', 'llm', 'hybrid') and first['best_score'] == 1.0
    assert first['final_route'] == 'jev' and first['triage_improvement']['status'] == 'skipped'
    assert (store.root / 'routed' / first['routed_id'] / 'best_spec.json').is_file()
    second = run_routed(TASK, TRAIN, VAL, jev=JevClient(mock=True), llm=LLMClient(kind='mock'), proposer=proposer, store=store,
                        evaluator=label_evaluator, task_id='sizes', route='hybrid', rounds=1, batch_size=4)
    assert second['route_source'] == 'override' and second['route'] == 'hybrid'
    assert second['triage_improvement']['status'] in ('accepted', 'rejected')
    history = triage_store.history(store)
    assert len(history) == 2 and {h['target'] for h in history} <= {'settled', 'open_ended', 'hybrid'}
    assert len(triage_store.improvements(store)) == 2
