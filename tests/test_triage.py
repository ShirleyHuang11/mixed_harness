import copy

import pytest

from auto_jev.llm import LLMClient
from auto_jev.runtime import PipelineRuntime
from auto_jev.spec import validate_spec
from auto_jev.triage import DEFAULT_TRIAGE_QUESTIONS, ROUTES, probe, route_of_spec, seed_for_route, triage, validate_task
from tests.support import ScriptedJev

LABEL_TASK = {'name': 'intents', 'objective': 'Route a message to the right intent.',
              'observation_fields': ['text'],
              'action_space': {'kind': 'labels', 'labels': ['card_lost', 'refund', 'other']},
              'evaluator': {'kind': 'exact_match'}, 'decisions_per_episode': 1, 'latency_budget_ms': 800,
              'examples': [{'observation': {'text': 'I lost my card'}, 'expected': 'card_lost'},
                           {'observation': {'text': 'money back please'}, 'expected': 'refund'}]}

TEXT_TASK = {'name': 'reply', 'objective': 'Write a reply that satisfies the constraints.',
             'observation_fields': ['instruction', 'constraints'],
             'action_space': {'kind': 'open'}, 'evaluator': {'kind': 'code_check'},
             'decisions_per_episode': 1, 'latency_budget_ms': None,
             'output_schema': {'type': 'object', 'required': ['text'], 'properties': {'text': {'type': 'string', 'minLength': 1}}},
             'examples': [{'observation': {'instruction': 'Say hi in under twenty words to a customer who waited.', 'constraints': []},
                           'expected': 'Hello, and thank you for your patience today.'}]}


def test_validate_task_rejects_missing_fields():
    bad = copy.deepcopy(LABEL_TASK); bad.pop('examples')
    with pytest.raises(ValueError, match='examples'):
        validate_task(bad)
    with pytest.raises(ValueError):
        validate_task({**LABEL_TASK, 'action_space': {'kind': 'labels'}})


def test_probe_is_deterministic_code_evidence():
    evidence = probe(LABEL_TASK)
    assert evidence == {'declared_action_kind': 'labels', 'label_count': 3, 'catalog_present_in_examples': False,
                        'exact_validator': True, 'expected_kind': 'label', 'median_expected_tokens': 1,
                        'decisions_per_episode': 1, 'latency_budget_ms': 800, 'example_count': 2}
    text = probe(TEXT_TASK)
    assert text['expected_kind'] == 'short_text' and text['exact_validator'] is True and text['label_count'] is None


def test_triage_sends_evidence_and_maps_confident_answer():
    jev = ScriptedJev({'route': {'settled': 0.9, 'open_ended': 0.05, 'hybrid': 0.05}, 'code_computes_facts': 0.8, 'needs_multistep': 0.1})
    decision = triage(LABEL_TASK, jev)
    assert decision['route'] == 'settled' and decision['confidence'] == 0.9
    assert decision['probabilities'] == {'settled': 0.9, 'open_ended': 0.05, 'hybrid': 0.05}
    assert decision['evidence'] == probe(LABEL_TASK)
    sent = jev.requests[0]
    assert sent['state']['evidence'] == probe(LABEL_TASK) and len(sent['state']['examples']) == 2
    assert set(sent['questions']) == set(DEFAULT_TRIAGE_QUESTIONS)
    assert decision['questions'] == DEFAULT_TRIAGE_QUESTIONS


def test_ambiguous_triage_still_returns_argmax_with_low_confidence():
    jev = ScriptedJev({'route': {'settled': 0.1, 'open_ended': 0.5, 'hybrid': 0.4}})
    decision = triage(TEXT_TASK, jev)
    assert decision['route'] == 'open_ended' and decision['confidence'] == 0.5


def test_triage_rejects_unknown_route():
    class Broken(ScriptedJev):
        def judge(self, state, questions, model=None):
            answer = super().judge(state, questions, model)
            answer['answers']['route']['choice'] = 'teleport'
            return answer
    with pytest.raises(ValueError, match='route'):
        triage(LABEL_TASK, Broken())


@pytest.mark.parametrize('route', ROUTES)
def test_seed_templates_validate_and_report_route(route):
    task = LABEL_TASK if route == 'settled' else TEXT_TASK
    spec = validate_spec(seed_for_route(route, task))
    assert spec['version'] == 4
    assert route_of_spec(spec) == {'settled': 'jev', 'open_ended': 'llm', 'hybrid': 'hybrid'}[route]


def sandbox_ok():
    from auto_jev.python_nodes import PythonSandboxUnavailable, _sandbox_command
    try:
        _sandbox_command(); return True
    except PythonSandboxUnavailable:
        return False


@pytest.mark.skipif(not sandbox_ok(), reason='needs the Python sandbox')
def test_settled_seed_runs_and_returns_a_label():
    spec = seed_for_route('settled', LABEL_TASK)
    jev = ScriptedJev({'action': {'card_lost': 0.1, 'refund': 0.8, 'other': 0.1}})
    result = PipelineRuntime(spec, jev).run({'text': 'money back please', 'labels': LABEL_TASK['action_space']['labels']})
    assert result['output'] == 'refund'
    assert set(jev.requests[0]['questions']['action']['criteria']) == {'card_lost', 'refund', 'other'}


@pytest.mark.skipif(not sandbox_ok(), reason='needs the Python sandbox')
def test_open_ended_seed_runs_with_schema():
    spec = seed_for_route('open_ended', TEXT_TASK)
    llm = LLMClient(kind='mock', mock_responder=lambda p, s: '{"text": "Hello and thanks for waiting."}')
    result = PipelineRuntime(spec, ScriptedJev(), llm).run({'instruction': 'greet', 'constraints': []})
    assert result['output'] == {'text': 'Hello and thanks for waiting.'}
    assert 'greet' in result['trace'][1]['prompt']


@pytest.mark.skipif(not sandbox_ok(), reason='needs the Python sandbox')
def test_hybrid_seed_generates_then_selects():
    spec = seed_for_route('hybrid', TEXT_TASK)
    llm = LLMClient(kind='mock', mock_responder=lambda p, s: '{"candidates": ["short", "a much longer candidate reply", "medium reply"]}')
    jev = ScriptedJev({'best': {'c0': 0.1, 'c1': 0.2, 'c2': 0.7}})
    result = PipelineRuntime(spec, jev, llm).run({'instruction': 'greet', 'constraints': []})
    assert result['output'] == {'text': 'medium reply'}
    criteria = jev.requests[0]['questions']['best']['criteria']
    assert criteria == {'c0': 'short', 'c1': 'a much longer candidate reply', 'c2': 'medium reply'}
