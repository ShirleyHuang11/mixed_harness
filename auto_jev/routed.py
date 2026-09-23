"""Level 1 orchestration: probe → Jev triage → one seed → one run_evolution. Re-routing happens inside evolution."""
import copy
import time
import uuid

from .evolution import run_evolution
from .spec import spec_hash, validate_spec
from .storage import RunStore, atomic_write_json, now_iso
from .triage import ROUTES, route_of_spec, seed_for_route, triage, validate_task

SEED_ROUTE_OF = {'settled': 'jev', 'open_ended': 'llm', 'hybrid': 'hybrid'}


def _routed_dir(store, task_id):
    ident = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{task_id}-{uuid.uuid4().hex[:6]}"
    path = store.root / 'routed' / ident
    path.mkdir(parents=True, exist_ok=False)
    return ident, path


def run_routed_evolution(task, train, validation, *, jev, llm, proposer, store, evaluator, task_id, route=None,
                         seed_pipeline=None, budget=None, evolution_rounds=None, max_metric_calls=None,
                         reflection_batch_size=None, seed=0, task_context=None, task_contract=None, triage_questions=None):
    task = validate_task(task)
    if route is not None and route not in ROUTES:
        raise ValueError('route must be one of ' + ', '.join(ROUTES))
    store = store or RunStore()
    # triage_store hook (Task 9): questions default to the stored, improved version.
    decision = triage(task, jev, questions=triage_questions, jev_model=getattr(jev, 'model', 'typesafe-ai/jev'))
    route_source = 'override' if route is not None else 'triage'
    route = route or decision['route']
    routed_id, path = _routed_dir(store, task_id)
    atomic_write_json(path / 'decision.json', {**decision, 'route_source': route_source, 'chosen_route': route})
    spec = copy.deepcopy(seed_pipeline) if seed_pipeline is not None else seed_for_route(route, task)
    llm_model = getattr(llm, 'model', None)
    if llm_model and any(node.get('kind') == 'llm' for node in spec['nodes']):
        spec['llm_model'] = llm_model
    spec = validate_spec(spec)
    context = copy.deepcopy(task_context) if task_context else {}
    context.setdefault('objective', task['objective'])
    context.update(route=route, route_source=route_source,
                   triage={key: decision[key] for key in ('route', 'probabilities', 'confidence', 'evidence')})
    summary = run_evolution(train, validation, jev=jev, llm=llm, proposer=proposer, store=store,
                            run_name=f'{task_id}-{route}', evaluator=evaluator, task_id=task_id, costs={},
                            seed_pipeline=spec, task_context=context, task_contract=task_contract, budget=budget,
                            evolution_rounds=evolution_rounds, max_metric_calls=max_metric_calls,
                            reflection_batch_size=reflection_batch_size, seed=seed)
    best = store.get_candidate(summary['run_id'], summary['best_hash'])['spec']
    result = {**summary, 'routed_id': routed_id, 'task_id': task_id, 'created_at': now_iso(), 'decision': decision,
              'route': route, 'route_source': route_source, 'seed_hash': spec_hash(spec), 'seed_route': SEED_ROUTE_OF[route],
              'seed_llm_model': spec.get('llm_model'), 'final_route': route_of_spec(best), 'budget': budget}
    # triage_store hook (Task 9): record history and attempt one improvement here.
    atomic_write_json(path / 'summary.json', result)
    return result
