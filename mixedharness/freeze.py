"""Freeze a selected spec into a self-describing artifact and evaluate it later on the same terms.

A frozen artifact binds the spec, its hash, the package version, the sandbox interpreter identity, the
evaluator's source hash and the Jev/LLM models. `evaluate_frozen` refuses to run when the runtime identity
or the evaluator changed, so a number from a frozen harness always says what produced it.
"""
import hashlib
import inspect

from . import __version__
from .evolve import evaluate_spec
from .sandbox import runtime_identity
from .spec import spec_hash, validate_spec
from .store import now_iso


class FrozenMismatch(RuntimeError):
    pass


def evaluator_identity(evaluator):
    try:
        source = inspect.getsource(evaluator)
    except (OSError, TypeError):
        source = repr(evaluator)
    return {'name': getattr(evaluator, '__qualname__', str(evaluator)), 'module': getattr(evaluator, '__module__', None),
            'sha256': hashlib.sha256(source.encode('utf-8')).hexdigest()}


def freeze(spec, *, evaluator, jev, llm=None, run_id=None, notes=None):
    spec = validate_spec(spec)
    return {'kind': 'mixedharness.frozen/1', 'frozen_at': now_iso(), 'package_version': __version__,
            'spec': spec, 'spec_hash': spec_hash(spec), 'runtime': runtime_identity(),
            'evaluator': evaluator_identity(evaluator),
            'jev': {'model': spec['jev_model'], 'transport': getattr(jev, 'transport', None)},
            'llm': {'model': spec.get('llm_model'), 'transport': getattr(llm, 'kind', None)} if spec.get('llm_model') else None,
            'run_id': run_id, 'notes': notes}


def evaluate_frozen(artifact, episodes, *, evaluator, jev, llm=None, budget=None, strict=True):
    if artifact.get('kind') != 'mixedharness.frozen/1':
        raise FrozenMismatch('not a frozen artifact')
    spec = validate_spec(artifact['spec'])
    if spec_hash(spec) != artifact['spec_hash']:
        raise FrozenMismatch('spec hash does not match the artifact')
    if strict:
        if evaluator_identity(evaluator)['sha256'] != artifact['evaluator']['sha256']:
            raise FrozenMismatch('the evaluator source changed since freezing')
        now = runtime_identity()
        for key in ('platform', 'python', 'interpreter'):
            if now.get(key) != artifact['runtime'].get(key):
                raise FrozenMismatch(f'runtime {key} changed: {artifact["runtime"].get(key)} -> {now.get(key)}')
    return evaluate_spec(spec, episodes, jev=jev, llm=llm, evaluator=evaluator, budget=budget)
