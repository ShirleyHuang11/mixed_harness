import pytest

from examples.classify import adapter as classify
from examples.constrained_text import adapter as constrained
from auto_jev.runtime import PipelineExecutionError
from tests.support import ScriptedJev

CONST = {'version': 2, 'name': 'const', 'jev_model': 'typesafe-ai/jev', 'nodes': [], 'output': "'refund'"}


def test_classify_fixture_splits_are_disjoint_and_labelled():
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    assert len(train) == 24 and len(validation) == 16
    assert {e['id'] for e in train}.isdisjoint({e['id'] for e in validation})
    labels = set(task['action_space']['labels'])
    assert all(e['label'] in labels and e['labels'] == task['action_space']['labels'] for e in train + validation)


def test_classify_scores_exact_label_and_hides_label_from_observation():
    _, train = classify.load('train')
    episode = next(e for e in train if e['label'] == 'refund')
    result = classify.evaluate_episode(CONST, episode, ScriptedJev())
    assert result['score'] == 1.0 and result['episode_id'] == episode['id']
    assert 'label' not in result['traces'][0]['observation']
    other = next(e for e in train if e['label'] != 'refund')
    assert classify.evaluate_episode(CONST, other, ScriptedJev())['score'] == 0.0


def test_classify_non_label_output_is_a_candidate_error():
    _, train = classify.load('train')
    with pytest.raises(PipelineExecutionError) as caught:
        classify.evaluate_episode({**CONST, 'output': "'nonsense'"}, train[0], ScriptedJev())
    assert not getattr(caught.value.cause, 'task_infrastructure_error', False)


@pytest.mark.parametrize('text,constraints,expected', [
    ('one two three', [{'kind': 'max_words', 'value': 3}], 1.0),
    ('one two three four', [{'kind': 'max_words', 'value': 3}], 0.0),
    ('hello world', [{'kind': 'must_include', 'value': 'world'}, {'kind': 'must_not_include', 'value': 'bye'}], 1.0),
    ('hello', [{'kind': 'min_words', 'value': 2}, {'kind': 'ends_with', 'value': 'o'}], 0.5),
])
def test_constraint_scoring(text, constraints, expected):
    assert constrained.score_constraints(text, constraints) == expected


def test_constrained_evaluator_requires_text_object():
    _, train = constrained.load('train')
    ok = {**CONST, 'output': "{'text': 'thank you for waiting today'}"}
    result = constrained.evaluate_episode(ok, train[0], ScriptedJev())
    assert 0.0 <= result['score'] <= 1.0 and result['output'] == {'text': 'thank you for waiting today'}
    with pytest.raises(PipelineExecutionError):
        constrained.evaluate_episode({**CONST, 'output': "'bare string'"}, train[0], ScriptedJev())


def test_unknown_constraint_kind_is_rejected():
    with pytest.raises(ValueError):
        constrained.score_constraints('x', [{'kind': 'rhymes', 'value': 'y'}])
