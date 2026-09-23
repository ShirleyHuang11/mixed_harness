"""Open-ended reply generation scored by deterministic constraint checks."""
import copy
import json
from pathlib import Path

from auto_jev.runtime import PipelineExecutionError, PipelineRuntime

HERE = Path(__file__).resolve().parent
TRAIN_ROWS = 12
KINDS = ('max_words', 'min_words', 'must_include', 'must_not_include', 'ends_with')


def load(split):
    task = json.loads((HERE / 'task.json').read_text())
    rows = json.loads((HERE / 'fixture.json').read_text())
    if split == 'train':
        rows = rows[:TRAIN_ROWS]
    elif split == 'validation':
        rows = rows[TRAIN_ROWS:]
    else:
        raise ValueError('split must be train or validation')
    return task, rows


def score_constraints(text, constraints):
    """Fraction of constraints satisfied; unknown kinds are a configuration error."""
    if not constraints:
        return 1.0
    words = text.split()
    passed = 0
    for constraint in constraints:
        kind, value = constraint['kind'], constraint['value']
        if kind not in KINDS:
            raise ValueError(f'unknown constraint kind {kind!r}')
        if kind == 'max_words':
            passed += len(words) <= value
        elif kind == 'min_words':
            passed += len(words) >= value
        elif kind == 'must_include':
            passed += str(value).lower() in text.lower()
        elif kind == 'must_not_include':
            passed += str(value).lower() not in text.lower()
        else:
            passed += text.rstrip().endswith(str(value))
    return passed / len(constraints)


def evaluate_episode(spec, episode, jev, llm=None, *, capture_traces=True):
    observation = {'instruction': episode['instruction'], 'constraints': copy.deepcopy(episode['constraints'])}
    runtime = PipelineRuntime(spec, jev, llm)
    try:
        decision = runtime.run(observation, memory={})
    except PipelineExecutionError as exc:
        partial = copy.deepcopy(exc.partial_result)
        partial.update(observation=observation, status='error')
        raise PipelineExecutionError(str(exc), partial_result={'episode_id': episode['id'], 'score': None,
                                     'error': str(exc), 'traces': [partial]}, cause=exc.cause, causes=exc.causes) from exc
    output = decision['output']
    record = {**decision, 'observation': observation, 'status': 'ok'}
    if not isinstance(output, dict) or not isinstance(output.get('text'), str) or not output['text'].strip():
        record.update(status='error', error='Policy must return {"text": nonempty string}')
        error = ValueError('Policy must return {"text": nonempty string}')
        raise PipelineExecutionError(str(error), partial_result={'episode_id': episode['id'], 'score': None,
                                     'error': str(error), 'traces': [record]}, cause=error) from error
    return {'episode_id': episode['id'], 'task_id': 'constrained_reply', 'status': 'completed',
            'score': score_constraints(output['text'], episode['constraints']), 'output': output,
            'traces': [record] if capture_traces else []}
