"""Budget, evolution loop, generic seeds, freeze. Mock providers only."""
import json

import pytest

from mixedharness import sandbox as sb
from mixedharness.budget import BudgetError, check_budget, episode_usage, validate_budget
from mixedharness.evolve import FAILURE_SCORE, EvolutionAborted, evaluate_spec, run_evolution
from mixedharness.freeze import FrozenMismatch, evaluate_frozen, freeze
from mixedharness.jev import JevClient
from mixedharness.llm import LLMClient
from mixedharness.reflect import Proposer, ReflectionError, build_prompt
from mixedharness.runtime import PipelineError, Runtime
from mixedharness.spec import FORMAT, spec_hash
from mixedharness.store import RunStore
from mixedharness.seeds import TaskError, route_of_spec, seed_for_route, validate_task

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


def test_evolution_aborts_on_infrastructure_failure(tmp_path, monkeypatch):
    for name in ('TYPESAFE_API_KEY', 'TYPESAFE_API_KEY_PERSONAL', 'TYPESAFE_API_KEY_personal'):   # another test may have loaded .env into the process
        monkeypatch.delenv(name, raising=False)
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


# -- generic seeds ----------------------------------------------------------------------------------------
TASK = {'name': 'sizes', 'objective': 'Say whether a number is big or small.', 'observation_fields': ['x'],
        'action_space': {'kind': 'labels', 'labels': LABELS}, 'evaluator': {'kind': 'exact_match'}, 'decisions_per_episode': 1,
        'latency_budget_ms': 1000, 'examples': [{'observation': {'x': 9}, 'expected': 'big'}, {'observation': {'x': 1}, 'expected': 'small'}]}


def test_task_validation():
    assert validate_task(TASK) == TASK
    with pytest.raises(TaskError):
        validate_task(dict(TASK, action_space={'kind': 'labels'}))
    with pytest.raises(TaskError):
        seed_for_route('hybrid', TASK)


@pytest.mark.parametrize('route,kinds', [('jev', 'jev'), ('llm', 'llm')])
def test_seed_templates_run_under_mock_providers(route, kinds):
    task = dict(TASK)
    if route == 'llm':
        task = dict(task, action_space={'kind': 'open'}, output_schema={'type': 'object', 'required': ['text'], 'properties': {'text': {'type': 'string'}}})
    spec = seed_for_route(route, task)
    assert route_of_spec(spec) == kinds
    rt = Runtime(spec, JevClient(mock=True), LLMClient(kind='mock'))
    try:
        result = rt.run({'x': 7, 'labels': LABELS})
    finally:
        rt.close()
    assert result['status'] == 'ok' and result['output'] is not None


def test_route_of_spec_reads_node_kinds():
    assert route_of_spec(SEED) == 'jev' and route_of_spec(BETTER) == 'code'
    mixed = dict(SEED, nodes=SEED['nodes'] + [{'id': 'd', 'kind': 'llm', 'prompt': "'x'"}])
    assert route_of_spec(mixed) == 'jev+llm'
