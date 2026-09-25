"""The routed run: probe → Jev triage → one seed → one evolution → record → improve the triage.

    summary = run_routed(task, train, validation, jev=..., llm=..., proposer=..., store=..., evaluator=...,
                         task_id='classify', route=None, rounds=3, batch_size=2, budget=None)

`route=` overrides the triage. Re-routing happens inside evolution, where the reflection model may swap node
kinds; `final_route` in the summary is what the selected harness ended up as, `route` is what the triage
chose. Artifacts: `<store>/routed/<id>/decision.json` and `summary.json`, plus the evolution run itself.
"""
import copy
import time
import uuid

from . import triage_store
from .evolve import run_evolution
from .spec import spec_hash, validate_spec
from .store import RunStore, atomic_write_json, now_iso
from .triage import ROUTES, TaskError, route_of_spec, seed_for_route, triage, validate_task

SEED_ROUTE_OF = {'jev': 'jev', 'llm': 'llm'}


def _routed_dir(store, task_id):
    ident = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{task_id}-{uuid.uuid4().hex[:6]}"
    path = store.root / 'routed' / ident
    path.mkdir(parents=True, exist_ok=False)
    return ident, path


def run_routed(task, train, validation, *, jev, llm, proposer, store, evaluator, task_id, route=None, seed_spec=None,
               budget=None, rounds=None, max_metric_calls=None, batch_size=None, seed=0, task_context=None,
               triage_questions=None, improve_triage=True, evaluator_kwargs=None):
    task = validate_task(task)
    if route is not None and route not in ROUTES:
        raise TaskError('route must be one of ' + ', '.join(ROUTES))
    store = store or RunStore()
    stored_questions, questions_version = triage_store.current_questions(store)
    decision = triage(task, jev, questions=triage_questions or stored_questions)
    route_source = 'override' if route is not None else 'triage'
    route = route or decision['route']
    routed_id, path = _routed_dir(store, task_id)
    atomic_write_json(path / 'decision.json', {**decision, 'route_source': route_source, 'chosen_route': route})
    jev_model = getattr(jev, 'model', None) or 'jev-1.13.0'
    llm_model = getattr(llm, 'model', None) if llm is not None else None
    if seed_spec is not None:
        spec = copy.deepcopy(seed_spec)
    else:
        spec = seed_for_route(route, task, jev_model=jev_model, llm_model=llm_model or 'mock-llm')
    spec['jev_model'] = jev_model
    if llm_model:
        spec['llm_model'] = llm_model   # pinned even for a settled seed so evolution may grow llm nodes
    spec = validate_spec(spec)
    context = dict(copy.deepcopy(task_context) if task_context else {})
    context.setdefault('id', task_id)
    context.setdefault('objective', task['objective'])
    context.update(route=route, route_source=route_source,
                   triage={k: decision[k] for k in ('route', 'probabilities', 'confidence', 'evidence')},
                   task_description={k: task[k] for k in ('name', 'observation_fields', 'action_space', 'evaluator', 'decisions_per_episode')})
    summary = run_evolution(train, validation, jev=jev, llm=llm, proposer=proposer, evaluator=evaluator, store=store,
                            seed_spec=spec, task=context, rounds=rounds, batch_size=batch_size, max_metric_calls=max_metric_calls,
                            budget=budget, seed=seed, run_name=f'{task_id}-{route}', evaluator_kwargs=evaluator_kwargs)
    best = summary['best_spec']
    result = {**{k: v for k, v in summary.items() if k != 'best_spec'}, 'routed_id': routed_id, 'task_id': task_id,
              'created_at': now_iso(), 'decision': decision, 'route': route, 'route_source': route_source,
              'seed_hash': spec_hash(spec), 'seed_route': SEED_ROUTE_OF[route], 'seed_llm_model': spec.get('llm_model'),
              'final_route': route_of_spec(best), 'best_spec': best, 'budget': budget,
              'questions_version': questions_version if triage_questions is None else None, 'task': {'name': task['name']}}
    atomic_write_json(path / 'summary.json', {k: v for k, v in result.items() if k != 'best_spec'})
    atomic_write_json(path / 'best_spec.json', best)
    triage_store.record(store, result)
    if improve_triage:
        try:
            result['triage_improvement'] = triage_store.improve(store, jev, proposer)
        except Exception as exc:   # an improvement failure never loses a completed run
            result['triage_improvement'] = {'status': 'error', 'reason': f'{type(exc).__name__}: {exc}'[:500]}
        atomic_write_json(path / 'summary.json', {k: v for k, v in result.items() if k != 'best_spec'})
    return result
