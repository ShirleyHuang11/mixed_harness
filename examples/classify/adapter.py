"""Single-step label routing. The label never enters the observation; the labels list does."""
import json
from pathlib import Path

from mixedharness.runtime import PipelineError, Runtime

HERE = Path(__file__).resolve().parent
TRAIN_ROWS = 24


def load(split):
    """Return (task, episodes) for 'train' (first 24 rows) or 'validation' (remaining 16)."""
    task = json.loads((HERE / 'task.json').read_text())
    rows = json.loads((HERE / 'fixture.json').read_text())
    if split == 'train':
        rows = rows[:TRAIN_ROWS]
    elif split == 'validation':
        rows = rows[TRAIN_ROWS:]
    else:
        raise ValueError('split must be train or validation')
    labels = list(task['action_space']['labels'])
    return task, [{**row, 'labels': labels} for row in rows]


def evaluate_episode(spec, episode, jev, llm=None, *, capture_traces=True):
    observation = {'text': episode['text'], 'labels': list(episode['labels'])}
    runtime = Runtime(spec, jev, llm)
    try:
        decision = runtime.run(observation, memory={})
    except PipelineError as exc:
        partial = dict(exc.partial, observation=observation, status='error')
        raise PipelineError(str(exc), partial={'episode_id': episode['id'], 'score': None, 'error': str(exc), 'traces': [partial]},
                            causes=exc.causes) from exc
    finally:
        runtime.close()
    output = decision['output']
    record = {**decision, 'observation': observation, 'status': 'ok'}
    if not isinstance(output, str) or output not in episode['labels']:
        record.update(status='error', error='the harness returned a value outside the label set')
        raise PipelineError('the harness returned a value outside the label set',
                            partial={'episode_id': episode['id'], 'score': None, 'error': record['error'], 'traces': [record]},
                            causes=[ValueError(record['error'])])
    return {'episode_id': episode['id'], 'task_id': 'banking_intents', 'status': 'completed',
            'score': 1.0 if output == episode['label'] else 0.0, 'output': output,
            'traces': [record] if capture_traces else []}
