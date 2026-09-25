"""tau2-bench retail as a Mixed Harness task: episodes are tau2 tasks, the evaluator runs one tau2 simulation.

`load(split)` returns (TaskDescription, episodes). Splits map to tau2's own: 'train' (74), 'validation'
(tau2 'test', 40), 'base' (all 114). Episode ids are unique across splits because tau2 task ids are.

`evaluate_episode(spec, episode, jev, llm=None, ...)` builds the environment, a `PipelineAgent` for the spec,
the official user simulator, and the orchestrator; runs the simulation with full evaluation; and returns the
evaluator contract: score (tau2 reward, 0 or 1), output (termination, reward breakdown, tau2 costs) and
one trace per agent turn. Infrastructure errors propagate as PipelineError with the partial trace.

tau2-bench must be importable (see docs/guides/tau2.md for the sibling checkout and `TAU2_HOME`).
"""
import copy
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_USER_LLM = 'gpt-4.1-2025-04-14'   # tau2 default; identical across arms
DEFAULT_MAX_STEPS = 60
SPLITS = {'train': 'train', 'validation': 'test', 'base': 'base'}


def ensure_tau2():
    try:
        import tau2  # noqa: F401
        return
    except ImportError:
        pass
    home = os.environ.get('TAU2_HOME') or str(Path.home() / 'tau2-bench')
    src = Path(home) / 'src'
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import tau2  # noqa: F401


def load(split, domain='retail'):
    ensure_tau2()
    from tau2.runner import get_tasks
    if split not in SPLITS:
        raise ValueError('split must be train, validation or base')
    task = json.loads((HERE / 'task.json').read_text())
    task['name'] = f'tau2_{domain}'
    tasks = get_tasks(domain, task_split_name=SPLITS[split])
    episodes = [{'id': f'{domain}-{t.id}', 'domain': domain, 'task_id': str(t.id), 'split': split} for t in tasks]
    return task, episodes


def _reward_summary(reward_info):
    if reward_info is None:
        return None
    data = reward_info.model_dump() if hasattr(reward_info, 'model_dump') else dict(reward_info)
    out = {'reward': data.get('reward'), 'reward_basis': [str(b) for b in (data.get('reward_basis') or [])]}
    for key in ('db_check', 'action_checks', 'env_assertions', 'nl_assertions', 'communicate_checks'):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            out[key] = {k: v for k, v in value.items() if not isinstance(v, (list, dict)) or k in ('db_match',)}
        elif isinstance(value, list):
            out[key] = [{k: v for k, v in (item.items() if isinstance(item, dict) else {}) if k in ('action_match', 'met', 'name', 'action_id', 'nl_assertion', 'info_match', 'justification')} for item in value]
        else:
            out[key] = value
    return out


def run_episode(spec, episode, jev, llm=None, *, user_llm=DEFAULT_USER_LLM, max_steps=DEFAULT_MAX_STEPS,
                seed=None, label='pipeline', user_llm_args=None):
    """Run one tau2 simulation with a PipelineAgent; returns (simulation_run, agent)."""
    ensure_tau2()
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner import build_environment, build_user, get_tasks, run_simulation
    from .agent import PipelineAgent
    domain = episode['domain']
    tasks = get_tasks(domain, task_ids=[episode['task_id']])
    if not tasks:
        raise ValueError(f"tau2 task {episode['task_id']} not found in domain {domain}")
    task = tasks[0]
    env = build_environment(domain)
    agent = PipelineAgent(env.get_tools(), env.get_policy(), spec, jev, llm, label=label)
    user = build_user('user_simulator', env, task, llm=user_llm, llm_args=user_llm_args or {'temperature': 0.0})
    orchestrator = Orchestrator(domain=domain, agent=agent, user=user, environment=env, task=task,
                                max_steps=max_steps, max_errors=5, seed=seed if seed is not None else episode.get('seed', 300))
    result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
    return result, agent


def evaluate_episode(spec, episode, jev, llm=None, *, capture_traces=True, user_llm=DEFAULT_USER_LLM,
                     max_steps=DEFAULT_MAX_STEPS, label='pipeline'):
    from mixedharness.runtime import PipelineError
    agent = None
    try:
        result, agent = run_episode(spec, episode, jev, llm, user_llm=user_llm, max_steps=max_steps, label=label)
    except PipelineError as exc:
        partial = {'episode_id': episode['id'], 'score': None, 'error': str(exc),
                   'traces': [_trace(e) for e in (agent.ledger if agent else [])] + [dict(copy.deepcopy(exc.partial or {}), status='error')]}
        raise PipelineError(str(exc), partial=partial, causes=exc.causes) from exc
    reward = float(result.reward_info.reward) if result.reward_info is not None else 0.0
    output = {'termination_reason': str(result.termination_reason), 'reward': reward,
              'reward_info': _reward_summary(result.reward_info), 'messages': len(result.messages or []),
              'tau2_agent_cost': result.agent_cost, 'tau2_user_cost': result.user_cost, 'duration_s': result.duration,
              'agent': agent.summary(), 'simulation_id': result.id, 'seed': result.seed}
    traces = [_trace(e) for e in agent.ledger] if capture_traces else []
    return {'episode_id': episode['id'], 'task_id': f"tau2_{episode['domain']}", 'status': 'completed',
            'score': reward, 'output': output, 'traces': traces}


def _trace(entry):
    """Ledger entry -> decision record: the complete runtime result plus observation and action."""
    decision = entry.get('decision') or {}
    record = dict(decision) if isinstance(decision, dict) else {}
    record.update(observation=entry['observation'], action=entry['action'], status='error' if entry.get('error') else 'ok',
                  elapsed_ms=entry['elapsed_ms'], turn=entry['turn'], chosen=entry.get('chosen'), invalid=entry.get('invalid'),
                  usage={k: entry[k] for k in ('jev_calls', 'llm_calls', 'llm_skipped', 'python_runs', 'jev_input_tokens',
                                               'llm_input_tokens', 'llm_output_tokens', 'cost_usd', 'unpriced_calls')})
    if entry.get('error'):
        record['error'] = entry['error']
    return record
