"""Evolve a spec with GEPA: whole-spec proposals from complete traces, strict acceptance, Pareto parents.

    summary = run_evolution(train, validation, jev=..., proposer=..., evaluator=..., store=..., seed_spec=...,
                            task={...}, llm=None, rounds=5, batch_size=2, budget=None, seed=0)

The evaluator is task code: `evaluator(spec, episode, jev, llm=None, capture_traces=True)` returning
`{'episode_id', 'score', 'traces', ...}` with a finite score, larger is better, and one trace per decision
(the complete runtime result plus observation and action). It raises `PipelineError` for a failed run.

Per round GEPA picks a parent from the Pareto frontier over training instances, evaluates it on a batch,
hands the complete trajectories to the proposer, evaluates the proposal on the same batch, and keeps it only
on strict improvement, then scores it on the full validation set. The loop stops after `rounds` proposals,
or at `max_metric_calls` episode evaluations. Infrastructure errors (provider, sandbox, reflection model)
abort the run; candidate errors score `FAILURE_SCORE` and continue. Every evaluation, candidate, trace and
reflection is written to the store as it happens.
"""
import copy
import inspect
import json
import math
import random
import time

import gepa
from gepa.core.adapter import EvaluationBatch

from .budget import check_budget, episode_usage, validate_budget
from .jev import JevError
from .llm import LLMError
from .reflect import Proposer, ReflectionError, build_prompt, strip_fences
from .runtime import PipelineError
from .sandbox import SandboxUnavailable
from .spec import SpecError, spec_hash, validate_spec
from .store import RunStore

FAILURE_SCORE = -1e9
INFRASTRUCTURE = (JevError, LLMError, SandboxUnavailable, ReflectionError)


class EvolutionAborted(RuntimeError):
    """An infrastructure failure stopped the run; the store holds everything up to that point."""


def _call_evaluator(evaluator, spec, episode, jev, llm, **kwargs):
    params = inspect.signature(evaluator).parameters
    accepts_llm = 'llm' in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_llm:
        return evaluator(spec, episode, jev, llm=llm, capture_traces=True, **kwargs)
    return evaluator(spec, episode, jev, capture_traces=True, **kwargs)


def evaluate_episode(evaluator, spec, episode, jev, llm=None, *, budget=None, evaluator_kwargs=None):
    """One evaluation with error classification: (row, result, fatal_exception_or_None)."""
    started = time.perf_counter()
    fatal = None
    try:
        result = _call_evaluator(evaluator, spec, episode, jev, llm, **(evaluator_kwargs or {}))
        score = result.get('score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('the evaluator returned a non-finite score')
        if score <= FAILURE_SCORE:
            raise ValueError('the evaluator returned a score at or below the failure sentinel')
        status = 'ok'
        usage = episode_usage(result)
        violations = check_budget(usage, budget) if budget else []
        if violations:
            status, score = 'over_budget', FAILURE_SCORE
            result = dict(result, score=score, error='over budget: ' + '; '.join(violations))
    except PipelineError as exc:
        status, score = 'error', FAILURE_SCORE
        result = dict(copy.deepcopy(exc.partial) if isinstance(exc.partial, dict) else {}, episode_id=episode['id'], score=score, error=str(exc))
        result.setdefault('traces', [])
        if exc.infrastructure:
            fatal = exc
    except INFRASTRUCTURE as exc:
        status, score = 'error', FAILURE_SCORE
        result = {'episode_id': episode['id'], 'score': score, 'error': str(exc), 'traces': []}
        fatal = exc
    except (ValueError, KeyError, IndexError, TypeError, ArithmeticError) as exc:
        status, score = 'error', FAILURE_SCORE
        result = {'episode_id': episode['id'], 'score': score, 'error': f'{type(exc).__name__}: {exc}', 'traces': []}
    row = {'episode_id': episode['id'], 'status': status, 'score': score, 'usage': episode_usage(result),
           'elapsed_ms': (time.perf_counter() - started) * 1000, 'error': result.get('error')}
    return row, result, fatal


def run_evolution(train, validation, *, jev, proposer, evaluator, store, seed_spec, task, llm=None, rounds=None,
                  batch_size=None, max_metric_calls=None, budget=None, seed=0, run_name='evolution',
                  max_prompt_bytes=None, evaluator_kwargs=None):
    if not train or not validation:
        raise ValueError('train and validation must be nonempty')
    ids = [e['id'] for e in train + validation]
    if len(set(ids)) != len(ids):
        raise ValueError('episode ids must be unique across train and validation')
    seed_spec = validate_spec(seed_spec)
    batch_size = len(train) if batch_size is None else int(batch_size)
    if not 1 <= batch_size <= len(train):
        raise ValueError('batch_size must be between 1 and the training set size')
    if rounds is not None and (isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1):
        raise ValueError('rounds must be a positive integer')
    if rounds is None and max_metric_calls is None:
        raise ValueError('give rounds or max_metric_calls')
    if budget is not None:
        budget = validate_budget(budget)
    if not isinstance(proposer, Proposer) and not callable(proposer):
        raise ValueError('proposer must be callable')
    store = store or RunStore()
    seed_hash = spec_hash(seed_spec)
    config = {'task': task, 'seed_spec_hash': seed_hash, 'rounds': rounds, 'batch_size': batch_size, 'max_metric_calls': max_metric_calls,
              'budget': budget, 'seed': seed, 'train': [e['id'] for e in train], 'validation': [e['id'] for e in validation],
              'jev': getattr(jev, 'metadata', None), 'llm': getattr(llm, 'metadata', None), 'proposer': getattr(proposer, 'metadata', None)}
    run_id = store.create_run(run_name, config)
    store.save_candidate(run_id, seed_spec, seed_hash, origin='seed')
    state = {'proposals': 0, 'accepted': 0, 'reflections': 0, 'abort': None, 'metric_calls': 0}
    spec_cache = {}

    def spec_of(candidate):
        text = candidate['spec']
        if text not in spec_cache:
            spec_cache[text] = validate_spec(json.loads(text))
        return spec_cache[text]

    class Adapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            if state['abort'] is not None:
                raise state['abort']
            spec = spec_of(candidate)
            ident = spec_hash(spec)
            store.save_candidate(run_id, spec, ident, origin='evaluated')
            outputs, scores, trajectories = [], [], []
            for item in batch:
                episode, split = item['episode'], item['split']
                row, result, fatal = evaluate_episode(evaluator, spec, episode, jev, llm, budget=budget, evaluator_kwargs=evaluator_kwargs)
                state['metric_calls'] += 1
                store.record_evaluation(run_id, dict(row, candidate=ident, split=split))
                if split == 'train' or row['status'] != 'ok':
                    store.save_trace(run_id, split, ident, episode['id'], result)
                if fatal is not None:
                    state['abort'] = EvolutionAborted(f'infrastructure failure on {episode["id"]}: {fatal}')
                    store.event(run_id, 'aborted', {'episode_id': episode['id'], 'error': str(fatal)})
                    raise state['abort']
                outputs.append({'episode_id': episode['id'], 'status': row['status'], 'score': row['score'], 'error': row['error']})
                scores.append(row['score'])
                if capture_traces:
                    trajectories.append({'episode': copy.deepcopy(episode), 'score': row['score'], 'status': row['status'], 'result': result})
            return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories if capture_traces else None)

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {'spec': list(eval_batch.trajectories or [])}

    def propose(candidate, reflective_dataset, components_to_update):
        if state['abort'] is not None:
            return dict(candidate)
        parent = spec_of(candidate)
        parent_hash = spec_hash(parent)
        trajectories = reflective_dataset.get('spec', [])
        state['reflections'] += 1
        index = state['reflections']
        try:
            prompt, payload = build_prompt(task, parent, trajectories, budget=budget, max_bytes=max_prompt_bytes or 1_000_000)
            store.save_reflection(run_id, index, payload)
            store.event(run_id, 'reflection', {'index': index, 'parent': parent_hash, 'bytes': payload['bytes'],
                                               'episodes': [t['episode']['id'] for t in trajectories]})
            text = proposer(prompt)
            store.save_reflection(run_id, index, payload, text)
        except ReflectionError as exc:
            state['abort'] = EvolutionAborted(f'reflection failed: {exc}')
            store.event(run_id, 'aborted', {'error': str(exc)})
            return dict(candidate)
        try:
            proposed = validate_spec(json.loads(strip_fences(text)))
            for key in ('format', 'jev_model'):
                if proposed.get(key) != parent.get(key):
                    raise SpecError(f'{key} may not change')
            if proposed.get('llm_model') != parent.get('llm_model'):
                raise SpecError('llm_model may not change')
        except (ValueError, SpecError) as exc:
            store.event(run_id, 'proposal_rejected', {'index': index, 'parent': parent_hash, 'reason': str(exc)[:500]})
            return dict(candidate)
        ident = spec_hash(proposed)
        store.save_candidate(run_id, proposed, ident, origin='proposal', parents=[parent_hash])
        store.event(run_id, 'proposal', {'index': index, 'parent': parent_hash, 'candidate': ident})
        state['proposals'] += 1
        return {'spec': json.dumps(proposed, sort_keys=True)}

    def stop(gepa_state):
        return rounds is not None and state['reflections'] >= rounds

    class QuietLogger:
        """GEPA's progress goes to the run directory, not to the console."""
        def __init__(self, path):
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)

        def log(self, message):
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(str(message) + '\n')

    store.update_run(run_id, status='running')
    try:
        result = gepa.optimize(seed_candidate={'spec': json.dumps(seed_spec, sort_keys=True)},
                               trainset=[{'split': 'train', 'episode': e} for e in train],
                               valset=[{'split': 'validation', 'episode': e} for e in validation],
                               adapter=Adapter(), custom_candidate_proposer=propose, reflection_lm=None,
                               candidate_selection_strategy='pareto', frontier_type='instance', module_selector='all',
                               skip_perfect_score=False, acceptance_criterion='strict_improvement',
                               val_evaluation_policy='full_eval', use_merge=False, cache_evaluation=False,
                               max_metric_calls=max_metric_calls, reflection_minibatch_size=batch_size,
                               stop_callbacks=[stop], seed=seed, run_dir=str(store.run_dir(run_id) / 'gepa'),
                               logger=QuietLogger(store.run_dir(run_id) / 'gepa' / 'run_log.txt'),
                               display_progress_bar=False, raise_on_exception=True)
    except EvolutionAborted:
        store.update_run(run_id, status='aborted', error=str(state['abort']))
        raise state['abort']
    if state['abort'] is not None:
        store.update_run(run_id, status='aborted', error=str(state['abort']))
        raise state['abort']
    candidates = [spec_of(c) for c in result.candidates]
    hashes = [spec_hash(c) for c in candidates]
    parents = [[hashes[j] for j in (ps or []) if j is not None] for ps in result.parents]
    val_scores = result.val_aggregate_scores
    for i, spec in enumerate(candidates):
        store.save_candidate(run_id, spec, hashes[i], origin='accepted', parents=parents[i], extra={'validation_score': val_scores[i]})
    valid = [i for i, sub in enumerate(result.val_subscores) if len(sub) == len(validation) and all(s > FAILURE_SCORE for s in sub.values())]
    if not valid:
        store.update_run(run_id, status='failed', error='no candidate passed every validation episode')
        raise EvolutionAborted('no candidate passed every validation episode')
    best = max(valid, key=lambda i: val_scores[i])
    summary = {'run_id': run_id, 'best_hash': hashes[best], 'best_spec': candidates[best], 'best_score': val_scores[best],
               'seed_score': val_scores[0], 'candidates': hashes, 'parents': parents, 'validation_scores': list(val_scores),
               'total_metric_calls': result.total_metric_calls, 'rounds': rounds, 'reflections': state['reflections'],
               'proposals_valid': state['proposals'], 'accepted': len(candidates) - 1,
               'jev_stats': copy.deepcopy(getattr(jev, 'stats', None)), 'llm_stats': copy.deepcopy(getattr(llm, 'stats', None)) if llm else None,
               'proposer_stats': copy.deepcopy(getattr(proposer, 'stats', None))}
    store.update_run(run_id, status='completed', result={k: v for k, v in summary.items() if k != 'best_spec'})
    return summary


def evaluate_spec(spec, episodes, *, jev, evaluator, llm=None, budget=None, evaluator_kwargs=None):
    """Score a fixed spec on episodes; returns {'mean', 'rows', 'results'} and raises on infrastructure failure."""
    spec = validate_spec(spec)
    rows, results = [], []
    for episode in episodes:
        row, result, fatal = evaluate_episode(evaluator, spec, episode, jev, llm, budget=budget, evaluator_kwargs=evaluator_kwargs)
        if fatal is not None:
            raise EvolutionAborted(f'infrastructure failure on {episode["id"]}: {fatal}')
        rows.append(row)
        results.append(result)
    scores = [r['score'] for r in rows if r['status'] == 'ok']
    return {'mean': (sum(scores) / len(scores)) if scores else None, 'completed': len(scores), 'expected': len(episodes),
            'rows': rows, 'results': results}
