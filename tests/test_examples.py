"""The bundled example tasks run end to end through seeding and evolution on mock providers."""
import json

import pytest

from examples.classify import adapter as classify
from examples.constrained_text import adapter as constrained
from mixedharness import sandbox as sb
from mixedharness.jev import JevClient
from mixedharness.llm import LLMClient
from mixedharness.reflect import Proposer
from mixedharness.evolve import run_evolution
from mixedharness.runtime import PipelineError
from mixedharness.store import RunStore
from mixedharness.seeds import route_of_spec, seed_for_route

pytestmark = pytest.mark.skipif(not sb.sandbox_available(), reason='no native sandbox on this machine')


def echo_parent(prompt):
    return json.dumps(json.loads(prompt)['parent_spec'])


def test_classify_probe_and_seed_evaluate():
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    assert len(train) == 24 and len(validation) == 16
    spec = seed_for_route('jev', task)
    result = classify.evaluate_episode(spec, train[0], JevClient(mock=True))
    assert result['score'] in (0.0, 1.0) and result['output'] in train[0]['labels'] and len(result['traces']) == 1


def test_constrained_text_seed_evaluates_and_scores_constraints():
    task, train = constrained.load('train')
    spec = seed_for_route('llm', task)
    responder = lambda prompt, schema: json.dumps({'text': 'We are sorry your delivery arrived late.'})
    result = constrained.evaluate_episode(spec, train[0], JevClient(mock=True), LLMClient(kind='mock', mock_responder=responder))
    assert result['score'] == 1.0
    assert constrained.score_constraints('a b c', [{'kind': 'max_words', 'value': 2}, {'kind': 'must_include', 'value': 'a'}]) == 0.5
    bad = LLMClient(kind='mock', mock_responder=lambda p, s: json.dumps({'text': ''}))
    with pytest.raises(PipelineError):
        constrained.evaluate_episode(spec, train[0], JevClient(mock=True), bad)


@pytest.mark.parametrize('module,task_id,route', [(classify, 'classify', 'jev'), (constrained, 'constrained_text', 'llm')])
def test_evolution_run_on_example(tmp_path, module, task_id, route):
    task, train = module.load('train')
    _, validation = module.load('validation')
    store = RunStore(tmp_path / 'runs')
    spec = seed_for_route(route, task, llm_model='mock-llm')
    spec['llm_model'] = 'mock-llm'
    summary = run_evolution(train[:4], validation[:4], jev=JevClient(mock=True), llm=LLMClient(kind='mock'),
                            proposer=Proposer({'kind': 'mock', 'responder': echo_parent}), store=store, seed_spec=spec,
                            evaluator=module.evaluate_episode, task={'id': task_id}, rounds=1, batch_size=2, run_name=task_id)
    assert summary['run_id'] and route_of_spec(summary['best_spec']) in ('jev', 'llm', 'jev+llm', 'code')
    assert (store.run_dir(summary['run_id']) / 'run.json').is_file()
    assert summary['total_metric_calls'] >= 4
