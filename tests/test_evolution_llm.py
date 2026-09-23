import json

import pytest

from auto_jev.evolution import FAILURE_SCORE, freeze_run, run_evolution
from auto_jev.frozen import evaluate_frozen
from auto_jev.llm import LLMClient
from auto_jev.runtime import PipelineRuntime
from auto_jev.storage import RunStore
from tests.support import ScriptedJev

SPEC = {'version': 4, 'name': 'echo', 'jev_model': 'typesafe-ai/jev', 'llm_model': 'mock-llm',
        'nodes': [{'id': 'draft', 'kind': 'llm', 'prompt_expression': "obs['text']",
                   'output_schema': {'type': 'object', 'required': ['label'], 'properties': {'label': {'type': 'string'}}}}],
        'output': "nodes['draft']['label']"}


def evaluate_with_llm(spec, episode, jev, llm=None, *, capture_traces=True):
    decision = PipelineRuntime(spec, jev, llm).run({'text': episode['text']})
    return {'episode_id': episode['id'], 'score': float(decision['output'] == episode['label']), 'traces': [decision]}


def evaluate_legacy(spec, episode, jev, capture_traces=True):
    return {'episode_id': episode['id'], 'score': 1.0, 'traces': [{'elapsed_ms': 5, 'trace': []}]}


def proposer(prompt):
    payload = json.loads(prompt)
    assert 'budget' in payload['task'] and set(payload['task']['budget']) == {'latency_ms_p95'}
    assert 'llm nodes' in payload['instruction']
    return json.dumps(payload['parent'])


def data():
    train = [{'id': f't{i}', 'text': 'x', 'label': 'x'} for i in range(2)]
    return train, [{'id': 'v0', 'text': 'x', 'label': 'x'}]


def test_llm_is_passed_to_evaluators_that_accept_it_and_frozen(tmp_path):
    train, validation = data()
    llm = LLMClient(kind='mock', mock_responder=lambda p, s: '{"label": "x"}')
    store = RunStore(tmp_path)
    summary = run_evolution(train, validation, jev=ScriptedJev(), llm=llm, proposer=proposer, store=store,
                            seed_pipeline=SPEC, evaluator=evaluate_with_llm, task_id='echo', costs={},
                            evolution_rounds=1, reflection_batch_size=1, budget={'latency_ms_p95': 5000})
    assert summary['best_score'] == 1.0
    assert store.get_run(summary['run_id'])['config']['budget'] == {'latency_ms_p95': 5000}
    artifact = freeze_run(store, summary['run_id'])
    assert artifact['budget'] == {'latency_ms_p95': 5000} and artifact['llm_metadata']['transport'] == 'mock'
    outputs = evaluate_frozen(artifact, validation, ScriptedJev(), evaluator=evaluate_with_llm, llm=llm)
    assert outputs[0]['score'] == 1.0
    with pytest.raises(ValueError, match='llm'):
        evaluate_frozen(artifact, validation, ScriptedJev(), evaluator=evaluate_with_llm)


def test_legacy_evaluator_without_llm_param(tmp_path):
    train, validation = data()
    summary = run_evolution(train, validation, jev=ScriptedJev(), llm=LLMClient(kind='mock'), proposer=proposer,
                            store=RunStore(tmp_path), seed_pipeline={**SPEC, 'nodes': [], 'output': "'x'"},
                            evaluator=evaluate_legacy, task_id='legacy', costs={}, evolution_rounds=1,
                            reflection_batch_size=1, budget={'latency_ms_p95': 5000})
    assert summary['best_score'] == 1.0


def test_over_budget_episode_is_invalid_and_recorded(tmp_path):
    train, validation = data()
    store = RunStore(tmp_path)
    with pytest.raises(ValueError, match='validation'):
        run_evolution(train, validation, jev=ScriptedJev(), llm=LLMClient(kind='mock'), proposer=proposer,
                      store=store, seed_pipeline={**SPEC, 'nodes': [], 'output': "'x'"}, evaluator=evaluate_legacy,
                      task_id='slow', costs={}, evolution_rounds=1, reflection_batch_size=1, budget={'latency_ms_p95': 1})
    run_id = store.list_runs()[0]['run_id']
    rows = store.list_evaluations(run_id, split='validation')
    assert rows[0]['status'] == 'over_budget' and rows[0]['score'] == FAILURE_SCORE
    assert rows[0]['usage']['latency_ms_p95'] == 5.0 and 'latency_ms_p95' in rows[0]['error']


def test_budget_not_applied_to_failed_episode(tmp_path):
    def broken(spec, episode, jev, capture_traces=True):
        raise ValueError('candidate produced an illegal action')
    train, validation = data()
    store = RunStore(tmp_path)
    with pytest.raises(ValueError):
        run_evolution(train, validation, jev=ScriptedJev(), proposer=proposer, store=store,
                      seed_pipeline={**SPEC, 'nodes': [], 'output': "'x'"}, evaluator=broken, task_id='broken',
                      costs={}, evolution_rounds=1, reflection_batch_size=1, budget={'latency_ms_p95': 0})
    run_id = store.list_runs()[0]['run_id']
    assert store.list_evaluations(run_id, split='validation')[0]['status'] == 'error'
