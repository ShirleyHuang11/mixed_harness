import json

import pytest

from auto_jev.evolution import freeze_run
from auto_jev.frozen import evaluate_frozen
from auto_jev.llm import LLMClient
from auto_jev.routed import run_routed_evolution
from auto_jev.storage import RunStore
from examples.classify import adapter as classify
from examples.constrained_text import adapter as constrained
from tests.support import ScriptedJev


def sandbox_ok():
    from auto_jev.python_nodes import PythonSandboxUnavailable, _sandbox_command
    try:
        _sandbox_command(); return True
    except PythonSandboxUnavailable:
        return False


pytestmark = pytest.mark.skipif(not sandbox_ok(), reason='seeds use Python nodes')


def echo_proposer(prompt):
    payload = json.loads(prompt)
    if 'current_questions' in payload:
        return json.dumps(payload['current_questions'])
    assert payload['task']['context']['route'] in ('settled', 'open_ended', 'hybrid')
    assert 'probabilities' in payload['task']['context']['triage'] and 'response' not in payload['task']['context']['triage']
    return json.dumps(payload['parent'])


def test_settled_task_routes_to_jev_runs_once_and_freezes(tmp_path):
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    jev = ScriptedJev({'route': {'settled': 0.9, 'open_ended': 0.05, 'hybrid': 0.05},
                       'action': {'card_lost': 0.2, 'refund': 0.6, 'balance': 0.1, 'transfer': 0.05, 'other': 0.05}})
    store = RunStore(tmp_path)
    summary = run_routed_evolution(task, train[:4], validation[:4], jev=jev, llm=LLMClient(kind='mock'),
                                   proposer=echo_proposer, store=store, evaluator=classify.evaluate_episode,
                                   task_id='banking', evolution_rounds=1, reflection_batch_size=1,
                                   budget={'latency_ms_p95': 60000})
    assert summary['decision']['route'] == 'settled' and summary['route'] == 'settled'
    assert summary['route_source'] == 'triage' and summary['seed_route'] == 'jev' and summary['final_route'] == 'jev'
    assert len(store.list_runs()) == 1 and summary['run_id'] == store.list_runs()[0]['run_id']
    assert store.get_run(summary['run_id'])['config']['task_context']['route'] == 'settled'
    saved = json.loads((tmp_path / 'routed' / summary['routed_id'] / 'summary.json').read_text())
    assert saved['run_id'] == summary['run_id'] and saved['decision']['probabilities']['settled'] == 0.9
    artifact = freeze_run(store, summary['run_id'])
    outputs = evaluate_frozen(artifact, validation[:2], jev, evaluator=classify.evaluate_episode, llm=LLMClient(kind='mock'))
    assert all(o['score'] in (0.0, 1.0) for o in outputs)


def test_open_ended_task_runs_llm_seed(tmp_path):
    task, train = constrained.load('train')
    _, validation = constrained.load('validation')
    jev = ScriptedJev({'route': {'settled': 0.1, 'open_ended': 0.5, 'hybrid': 0.4}})
    llm = LLMClient(kind='mock', model='mock-llm', mock_responder=lambda p, s: json.dumps({'text': 'We are sorry and thank you; your refund is issued.'}))
    summary = run_routed_evolution(task, train[:3], validation[:3], jev=jev, llm=llm, proposer=echo_proposer,
                                   store=RunStore(tmp_path), evaluator=constrained.evaluate_episode,
                                   task_id='reply', evolution_rounds=1, reflection_batch_size=1)
    assert summary['route'] == 'open_ended' and summary['decision']['confidence'] == 0.5
    assert summary['seed_route'] == 'llm' and summary['final_route'] == 'llm' and summary['seed_llm_model'] == 'mock-llm'
    assert 0.0 <= summary['best_score'] <= 1.0


def test_route_override_runs_hybrid_seed(tmp_path):
    task, train = constrained.load('train')
    _, validation = constrained.load('validation')
    jev = ScriptedJev({'route': {'settled': 0.1, 'open_ended': 0.5, 'hybrid': 0.4}, 'best': {'c0': 0.2, 'c1': 0.7, 'c2': 0.1}})

    def responder(prompt, schema):
        return json.dumps({'candidates': ['sorry', 'We are sorry, thank you, refund issued.', 'x']})

    summary = run_routed_evolution(task, train[:2], validation[:2], jev=jev, llm=LLMClient(kind='mock', mock_responder=responder),
                                   proposer=echo_proposer, store=RunStore(tmp_path), evaluator=constrained.evaluate_episode,
                                   task_id='reply', route='hybrid', evolution_rounds=1, reflection_batch_size=1)
    assert summary['decision']['route'] == 'open_ended' and summary['route'] == 'hybrid'
    assert summary['route_source'] == 'override' and summary['seed_route'] == 'hybrid' and summary['final_route'] == 'hybrid'


def test_seed_pipeline_override_replaces_template(tmp_path):
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    jev = ScriptedJev({'route': {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0}})
    override = {'version': 4, 'name': 'const', 'jev_model': 'typesafe-ai/jev', 'nodes': [], 'output': "'refund'"}
    summary = run_routed_evolution(task, train[:2], validation[:2], jev=jev, llm=LLMClient(kind='mock'),
                                   proposer=echo_proposer, store=RunStore(tmp_path), evaluator=classify.evaluate_episode,
                                   task_id='banking', evolution_rounds=1, reflection_batch_size=1, seed_pipeline=override)
    assert summary['seed_route'] == 'jev' and summary['final_route'] == 'code'


def test_unknown_route_override_is_rejected(tmp_path):
    task, train = classify.load('train')
    with pytest.raises(ValueError, match='route'):
        run_routed_evolution(task, train[:1], train[1:2], jev=ScriptedJev(), llm=LLMClient(kind='mock'),
                             proposer=echo_proposer, store=RunStore(tmp_path), evaluator=classify.evaluate_episode,
                             task_id='banking', route='teleport', evolution_rounds=1, reflection_batch_size=1)
