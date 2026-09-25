"""The bundled example tasks run end to end through the routed pipeline on mock providers."""
import json

import pytest

from examples.classify import adapter as classify
from examples.constrained_text import adapter as constrained
from mixedharness import sandbox as sb
from mixedharness.jev import JevClient
from mixedharness.llm import LLMClient
from mixedharness.reflect import Proposer
from mixedharness.routed import run_routed
from mixedharness.runtime import PipelineError
from mixedharness.store import RunStore
from mixedharness.triage import probe, seed_for_route

pytestmark = pytest.mark.skipif(not sb.sandbox_available(), reason='no native sandbox on this machine')


def echo_parent(prompt):
    return json.dumps(json.loads(prompt)['parent_spec'])


def test_classify_probe_and_seed_evaluate():
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    assert len(train) == 24 and len(validation) == 16 and probe(task)['expected_kind'] == 'label'
    spec = seed_for_route('settled', task)
    result = classify.evaluate_episode(spec, train[0], JevClient(mock=True))
    assert result['score'] in (0.0, 1.0) and result['output'] in train[0]['labels'] and len(result['traces']) == 1


def test_constrained_text_seed_evaluates_and_scores_constraints():
    task, train = constrained.load('train')
    spec = seed_for_route('open_ended', task)
    responder = lambda prompt, schema: json.dumps({'text': 'We are sorry your delivery arrived late.'})
    result = constrained.evaluate_episode(spec, train[0], JevClient(mock=True), LLMClient(kind='mock', mock_responder=responder))
    assert result['score'] == 1.0
    assert constrained.score_constraints('a b c', [{'kind': 'max_words', 'value': 2}, {'kind': 'must_include', 'value': 'a'}]) == 0.5
    bad = LLMClient(kind='mock', mock_responder=lambda p, s: json.dumps({'text': ''}))
    with pytest.raises(PipelineError):
        constrained.evaluate_episode(spec, train[0], JevClient(mock=True), bad)


@pytest.mark.parametrize('module,task_id,route', [(classify, 'classify', None), (constrained, 'constrained_text', 'hybrid')])
def test_routed_run_on_example(tmp_path, module, task_id, route):
    task, train = module.load('train')
    _, validation = module.load('validation')
    store = RunStore(tmp_path / 'runs')
    summary = run_routed(task, train[:4], validation[:4], jev=JevClient(mock=True), llm=LLMClient(kind='mock'),
                         proposer=Proposer({'kind': 'mock', 'responder': echo_parent}), store=store,
                         evaluator=module.evaluate_episode, task_id=task_id, route=route, rounds=1, batch_size=2)
    assert summary['run_id'] and summary['final_route'] in ('jev', 'llm', 'hybrid', 'code')
    assert (store.root / 'routed' / summary['routed_id'] / 'summary.json').is_file()
    assert summary['total_metric_calls'] >= 4
