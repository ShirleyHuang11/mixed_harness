"""Findings from the whole-branch review; each test failed before its fix."""
import json

import pytest

from auto_jev.evolution import run_evolution
from auto_jev.llm import LLMClient, LLMOutputError, LLMProviderError
from auto_jev.routed import run_routed_evolution
from auto_jev.runtime import PipelineRuntime
from auto_jev.schema import validate_json_schema
from auto_jev.spec import validate_spec
from auto_jev.storage import RunStore
from auto_jev.triage import seed_for_route
from examples.classify import adapter as classify
from tests.support import ScriptedJev


def sandbox_ok():
    from auto_jev.python_nodes import PythonSandboxUnavailable, _sandbox_command
    try:
        _sandbox_command(); return True
    except PythonSandboxUnavailable:
        return False


needs_sandbox = pytest.mark.skipif(not sandbox_ok(), reason='seeds use Python nodes')
ROUTE = {'route': {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0}}
ACTION = {'action': {'card_lost': 0.1, 'refund': 0.7, 'balance': 0.1, 'transfer': 0.05, 'other': 0.05}}


def add_llm_node(prompt):
    payload = json.loads(prompt)
    if 'current_questions' in payload:
        return json.dumps(payload['current_questions'])
    parent = payload['parent']
    assert payload['instruction'].startswith('Evolve') and 'Use version 4.' in payload['instruction']
    parent['nodes'].append({'id': 'note', 'kind': 'llm', 'prompt_expression': "obs['text']"})
    return json.dumps(parent)


@needs_sandbox
def test_settled_seed_can_evolve_an_llm_node_and_prompt_names_version_4(tmp_path):
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    store = RunStore(tmp_path)
    summary = run_routed_evolution(task, train[:2], validation[:2], jev=ScriptedJev({**ROUTE, **ACTION}), llm=LLMClient(kind='mock'),
                                   proposer=add_llm_node, store=store, evaluator=classify.evaluate_episode,
                                   task_id='banking', evolution_rounds=1, reflection_batch_size=1)
    assert not store.list_events(summary['run_id'], 'proposal_rejected')
    assert store.list_events(summary['run_id'], 'proposal')
    assert summary['seed_llm_model'] == 'mock-llm'


@pytest.mark.parametrize('schema', [{'enum': []}, {'type': 'object', 'properties': [], 'required': ['x']},
                                    {'type': 'array', 'minItems': 1, 'items': []}, {'type': 'string', 'minLength': 'x'}])
def test_malformed_output_schema_is_a_value_error(schema):
    with pytest.raises(ValueError):
        validate_json_schema({}, schema)
    with pytest.raises(ValueError):
        validate_spec({'version': 4, 'name': 'x', 'jev_model': 'j', 'llm_model': 'm', 'output': "nodes['d']",
                       'nodes': [{'id': 'd', 'kind': 'llm', 'prompt_expression': "'p'", 'output_schema': schema}]})


def test_llm_transport_failures_are_infrastructure_but_output_failures_are_not(tmp_path):
    with pytest.raises(LLMProviderError) as missing_key:
        LLMClient(kind='anthropic', api_key_env='NO_SUCH_KEY_VAR').complete('x')
    assert missing_key.value.task_infrastructure_error is True
    replay = LLMClient(kind='mock', cache_dir=tmp_path, cache_namespace='ns', cache_only=True)
    with pytest.raises(LLMProviderError) as miss:
        replay.complete('never cached')
    assert miss.value.task_infrastructure_error is True
    bad = LLMClient(kind='mock', mock_responder=lambda p, s: '{"label": "zzz"}')
    with pytest.raises(LLMOutputError) as output:
        bad.complete('x', output_schema={'type': 'object', 'required': ['label'], 'properties': {'label': {'enum': ['a']}}})
    assert isinstance(output.value, ValueError) and not getattr(output.value, 'task_infrastructure_error', False)


def test_llm_transport_failure_stops_evolution_instead_of_scoring(tmp_path):
    spec = {'version': 4, 'name': 'x', 'jev_model': 'j', 'llm_model': 'm', 'output': "nodes['d']",
            'nodes': [{'id': 'd', 'kind': 'llm', 'prompt_expression': "obs['text']"}]}

    def evaluate(spec, episode, jev, llm=None, *, capture_traces=True):
        decision = PipelineRuntime(spec, jev, llm).run({'text': episode['text']})
        return {'episode_id': episode['id'], 'score': 1.0, 'traces': [decision]}

    llm = LLMClient(kind='mock', cache_dir=tmp_path, cache_namespace='ns', cache_only=True)
    with pytest.raises(LLMProviderError):
        run_evolution([{'id': 't', 'text': 'a'}], [{'id': 'v', 'text': 'b'}], jev=ScriptedJev(), llm=llm,
                      proposer=lambda p: p, store=RunStore(tmp_path), seed_pipeline=spec, evaluator=evaluate,
                      task_id='x', costs={}, evolution_rounds=1, reflection_batch_size=1)


@needs_sandbox
def test_settled_seed_falls_back_to_task_labels_when_obs_has_none():
    task, _ = classify.load('train')
    task = {**task, 'observation_fields': ['text']}
    spec = seed_for_route('settled', task)
    jev = ScriptedJev(ACTION)
    assert PipelineRuntime(spec, jev).run({'text': 'money back'})['output'] == 'refund'
    assert set(jev.requests[0]['questions']['action']['criteria']) == set(task['action_space']['labels'])


@needs_sandbox
def test_jev_failure_during_triage_improvement_does_not_lose_the_run(tmp_path):
    from auto_jev import triage_store

    class FlakyJev(ScriptedJev):
        def judge(self, state, questions, model=None):
            if 'history' in state or state.get('marker') == 'replay':
                raise RuntimeError('jev down')
            return super().judge(state, questions, model)

    task, train = classify.load('train')
    _, validation = classify.load('validation')
    store = RunStore(tmp_path)
    triage_store.record(store, {'routed_id': 'old', 'task_id': 'old', 'run_id': 'r', 'best_score': 1.0, 'route': 'settled',
                                'route_source': 'triage', 'final_route': 'jev', 'questions_version': 0, 'task': {'name': 'old'},
                                'decision': {'route': 'settled', 'probabilities': {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0},
                                             'state': {'marker': 'replay'}}})
    echo = lambda p: json.dumps(json.loads(p).get('current_questions') or json.loads(p)['parent'])
    summary = run_routed_evolution(task, train[:2], validation[:2], jev=FlakyJev({**ROUTE, **ACTION}), llm=LLMClient(kind='mock'),
                                   proposer=echo, store=store, evaluator=classify.evaluate_episode,
                                   task_id='banking', evolution_rounds=1, reflection_batch_size=1)
    assert summary['triage_improvement']['status'] == 'error' and 'jev down' in summary['triage_improvement']['reason']
    saved = json.loads((tmp_path / 'routed' / summary['routed_id'] / 'summary.json').read_text())
    assert saved['run_id'] == summary['run_id'] and saved['triage_improvement']['status'] == 'error'
