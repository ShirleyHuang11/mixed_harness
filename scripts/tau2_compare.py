"""The arms on tau2-bench, same environment, same user simulator, same tasks and seeds.

Arms
  llm_only        tau2's stock `llm_agent` on --agent-llm (litellm name): every step escalates, one full model call
                  per turn, no Jev. The benchmark's own reference agent; the positive control.
  jev_only        examples.tau2.seeds.jev_only_seed(): every step on the closed path. Compiler, one Jev request, code
                  fills arguments, templates reply. No language model. Jev's route pick is recorded, never acted on.
  routed          examples.tau2.seeds.routed_seed(policy=...): the table from --policy says who takes each step: a code
                  row takes it with no model call, Jev's pick is kept when its probability clears the cluster's threshold,
                  otherwise the LLM takes the step (one native tool-calling turn).
  routed_evolved  a spec produced by evolution of the routed seed (--evolved-spec <path>); without one, the routed
                  seed with a loop-B policy fitted on --policy-steps (step rows from an earlier run), or the plain
                  routed seed again when neither is given (the arm then duplicates `routed`, and the table says so).

Decision table (evaluated after the run; stage 2 of docs/design/tau2-four-arms.md). h = the 95% half-width on pass^1
at n tasks (±7.5 points at n=114).
  llm_only pass^1 < 0.60                                     -> STOP: positive control failed; user sim / env suspect
  jev_only llm_calls > 0 or llm_only jev_calls > 0           -> STOP: instrument broken
  routed within h of llm_only and agent cost <= 0.5x         -> HOLDS: the claim as pre-registered
  routed within h of llm_only, cost > 0.5x                   -> INCONCLUSIVE ON COST: report the cost ratio, no claim
  routed more than h below the best single arm               -> FAILS
  routed_evolved vs routed                                   -> reported, no rule: it is scored on the held-out half

Run inside the tau2-bench virtualenv with this repo installed into it (docs/guides/tau2.md):
  ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --num-tasks 5
  ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --arms jev_only,routed --mock --num-tasks 2 \
      --user replay --replay-results runs/tau2/<stamp>/llm_only/results.json                             # plumbing, no credit

Outputs runs/tau2/<stamp>/{manifest.json, summary.json, table.md, <arm>/results.json, <arm>/ledgers.json,
<arm>/steps.jsonl}. steps.jsonl is the step table of the trace store (one row per agent turn: cluster, closed flag,
Jev's route pick, the realised path and why, agreement, confidence, action, guard, cost, latency, episode reward).
table.md is generated from the artifacts, never typed. With --trials > 1 the reward joined to a step row is the
reward of the k-th simulation of that task in results order, which matches the k-th agent when concurrency is 1.
"""
import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.tau2 import adapter  # noqa: E402
import importlib  # noqa: E402

SEEDS, routed_seed = None, None   # bound by _load_seeds (default examples.tau2.seeds; --seed-module picks another version)


def _load_seeds(name):
    global SEEDS, routed_seed
    mod = importlib.import_module(name)
    SEEDS, routed_seed = mod.SEEDS, mod.routed_seed
    return mod
from mixedharness.jev import JevClient  # noqa: E402
from mixedharness.llm import LLMClient  # noqa: E402
from mixedharness.router import fit_policy, route_metrics  # noqa: E402
from mixedharness.sandbox import WorkerPool  # noqa: E402
from mixedharness.trace import read_steps, write_steps  # noqa: E402

FLUSH_SECONDS = 60   # ledgers and steps of finished-so-far agents are written this often, so a killed run keeps them

ARMS = ('llm_only', 'jev_only', 'routed', 'routed_evolved')
PIPELINE_ARMS = ('jev_only', 'routed', 'routed_evolved')
NO_TEMPERATURE_MODELS = ('claude-sonnet-5', 'claude-opus-5', 'claude-fable')


def llm_args_for(model):
    """temperature=0 wherever the provider still accepts it; the Claude 5 family rejects the parameter. Through the local
    subscription proxy (OPENAI_API_BASE set) the base URL and a placeholder key travel in the args as well."""
    args = {} if any(tag in (model or '') for tag in NO_TEMPERATURE_MODELS) else {'temperature': 0.0}
    if (model or '').startswith('openai/') and os.environ.get('OPENAI_API_BASE'):
        args.update({'api_base': os.environ['OPENAI_API_BASE'], 'api_key': os.environ.get('OPENAI_API_KEY', 'subscription')})
        args.pop('temperature', None)
    return args


class _LenientJson:
    """tau2's NL-assertion evaluator does `json.loads(reply)` on the judge's raw text. GPT-4.1 returns bare JSON;
    Claude wraps it in a code fence or a sentence, which made 40 of 114 simulations fail at evaluation time
    (JSONDecodeError, recorded as an infrastructure error after the conversation had run and been paid for).
    This shim extracts the first JSON document from the reply before decoding."""
    dumps = staticmethod(json.dumps)

    @staticmethod
    def loads(text, *args, **kwargs):
        from mixedharness.llm import extract_json
        try:
            return json.loads(text, *args, **kwargs)
        except (ValueError, TypeError):
            return json.loads(extract_json(str(text)))


def set_nl_judge(model):
    """tau2 reads the NL-assertion judge from a module constant; override it in place and return what was there."""
    import tau2.evaluator.evaluator_nl_assertions as nl
    previous = (nl.DEFAULT_LLM_NL_ASSERTIONS, dict(nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS))
    nl.DEFAULT_LLM_NL_ASSERTIONS = model
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = llm_args_for(model)
    nl.json = _LenientJson
    return previous


def git_head(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int(round(fraction * len(ordered) + 0.5)) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def half_width(n, p=0.8):
    return 1.96 * (p * (1 - p) / n) ** 0.5 * 100 if n else None


def summarize_results(results, agents_by_sim=None):
    """Per-arm metrics from a tau2 Results object (+ our ledgers for pipeline arms)."""
    from tau2.data_model.message import AssistantMessage
    all_sims = list(results.simulations or [])
    # A simulation without a reward never ran to evaluation (infrastructure error, retries exhausted).
    # It is reported as failed, not as a zero: a missing reading is not a negative result.
    sims = [s for s in all_sims if s.reward_info is not None and s.reward_info.reward is not None]
    failed = [str(s.termination_reason) for s in all_sims if s not in sims]
    rewards, costs, user_costs, latencies, turns = [], [], [], [], []
    llm_in = llm_out = 0
    terminations = {}
    for sim in sims:
        rewards.append(float(sim.reward_info.reward))
        costs.append(sim.agent_cost)
        user_costs.append(sim.user_cost)
        terminations[str(sim.termination_reason)] = terminations.get(str(sim.termination_reason), 0) + 1
        n = 0
        for m in sim.messages or []:
            if isinstance(m, AssistantMessage):
                n += 1
                if m.generation_time_seconds is not None:
                    latencies.append(float(m.generation_time_seconds) * 1000)
                u = m.usage or {}
                llm_in += int(u.get('prompt_tokens', u.get('llm_input_tokens', 0)) or 0)
                llm_out += int(u.get('completion_tokens', u.get('llm_output_tokens', 0)) or 0)
        turns.append(n)
    priced = [c for c in costs if c is not None]
    out = {'completed': len(sims), 'failed': len(failed), 'failed_reasons': failed,
           'pass1': statistics.mean(rewards) if rewards else None,
           'successes': int(sum(1 for r in rewards if r >= 1.0)),
           'agent_cost_per_task_usd': statistics.mean(priced) if priced and len(priced) == len(costs) else None,
           'agent_cost_priced_fraction': (len(priced) / len(costs)) if costs else None,
           'user_cost_per_task_usd': statistics.mean([c for c in user_costs if c is not None]) if any(c is not None for c in user_costs) else None,
           'turns_per_task': statistics.mean(turns) if turns else None,
           'turn_latency_ms_median': percentile(latencies, 0.5), 'turn_latency_ms_p95': percentile(latencies, 0.95),
           'llm_input_tokens_per_turn': (llm_in / sum(turns)) if sum(turns) else None,
           'llm_output_tokens_per_turn': (llm_out / sum(turns)) if sum(turns) else None,
           'terminations': terminations}
    if agents_by_sim:
        totals = {}
        models = {'jev': set(), 'llm': set()}
        model_ms, sandbox_ms = [], []
        for agent in agents_by_sim.values():
            for k, v in agent.counters.items():
                totals[k] = totals.get(k, 0) + v
            for kind, ids in agent.actual_models.items():
                models[kind] |= set(ids)
            for entry in agent.ledger:
                model_ms.append(entry.get('model_ms', 0.0)); sandbox_ms.append(entry.get('sandbox_ms', 0.0))
        t = max(totals.get('turns', 0), 1)
        out.update({'turn_model_latency_ms_median': percentile(model_ms, 0.5), 'turn_sandbox_ms_median': percentile(sandbox_ms, 0.5),
                    'jev_calls_per_turn': totals.get('jev_calls', 0) / t, 'llm_calls_per_turn': totals.get('llm_calls', 0) / t,
                    'llm_skipped_per_turn': totals.get('llm_skipped', 0) / t, 'python_runs_per_turn': totals.get('python_runs', 0) / t,
                    'jev_input_tokens_per_turn': totals.get('jev_input_tokens', 0) / t,
                    'invalid_actions': totals.get('invalid_actions', 0), 'candidate_errors': totals.get('candidate_errors', 0),
                    'unpriced_calls': totals.get('unpriced_calls', 0), 'actual_models': {k: sorted(v) for k, v in models.items()},
                    'counters_total': totals})
    else:
        # Stock LLM agent: one model call per turn and no sandbox, so model latency is the turn latency.
        out.update({'jev_calls_per_turn': 0.0, 'llm_calls_per_turn': 1.0,
                    'turn_model_latency_ms_median': out['turn_latency_ms_median'], 'turn_sandbox_ms_median': 0.0})
    return out


def _completed_attempt(agent):
    """False for an agent whose attempt died on an infrastructure error (tau2 retries build a fresh agent)."""
    ledger = agent.ledger
    return bool(ledger) and not (ledger[-1].get('error') and 'infrastructure' in str(ledger[-1].get('error')).lower())


def _steps_with_rewards(results, agents, arm):
    """Join each completed agent attempt to the reward of the k-th simulation of its task, in results order.

    tau2's retry loop builds a new agent for every attempt of a task, so agents whose ledger ends in an
    infrastructure error are attempts that never produced a simulation and are skipped in the join."""
    sims_by_task = {}
    for sim in results.simulations or []:
        sims_by_task.setdefault(str(sim.task_id), []).append(sim)
    seen = {}
    rows = []
    for (task_id, _), agent in sorted(agents.items(), key=lambda kv: kv[0][1]):
        if not _completed_attempt(agent):
            rows.extend(agent.steps(episode=task_id, arm=arm, trial=-1, reward=None))   # kept, marked as a dead attempt
            continue
        k = seen.get(task_id, 0)
        seen[task_id] = k + 1
        sims = sims_by_task.get(task_id, [])
        sim = sims[k] if k < len(sims) else None
        reward = float(sim.reward_info.reward) if sim is not None and sim.reward_info is not None and sim.reward_info.reward is not None else None
        rows.extend(agent.steps(episode=task_id, arm=arm, trial=k, reward=reward))
    return rows


def _llm_only_steps(results, arm='llm_only'):
    """The stock agent has no ledger; its steps are its assistant turns, all on the escalate path."""
    from tau2.data_model.message import AssistantMessage
    rows = []
    seen = {}
    for sim in results.simulations or []:
        task_id = str(sim.task_id)
        k = seen.get(task_id, 0); seen[task_id] = k + 1
        reward = float(sim.reward_info.reward) if sim.reward_info is not None and sim.reward_info.reward is not None else None
        turn = 0
        for m in sim.messages or []:
            if isinstance(m, AssistantMessage):
                turn += 1
                call = m.tool_calls[0] if m.tool_calls else None
                rows.append({'episode': task_id, 'trial': k, 'turn': turn, 'arm': arm, 'spec_hash': None, 'harness_version': 'tau2 llm_agent',
                             'cluster': None, 'closed': None, 'needs': None, 'path_jev': None, 'path': 'escalate', 'path_source': 'forced',
                             'agreement': None, 'confidence': None, 'fits': None,
                             'action_kind': 'tool' if call else 'message', 'action_name': call.name if call else None, 'chosen': None,
                             'guard': None, 'invalid': None, 'jev_calls': 0, 'llm_calls': 1, 'cost_usd': m.cost,
                             'latency_ms': (m.generation_time_seconds or 0.0) * 1000, 'model_ms': None, 'reward': reward})
    return rows


def run_arm(arm, args, tasks, out_dir, jev, llm, evolved_spec):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.runner import run_tasks
    from examples.tau2.agent import PipelineAgent
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    agents = {}
    if arm == 'llm_only':
        agent_name, agent_llm = 'llm_agent', args.agent_llm
    else:
        if arm == 'routed_evolved':
            spec = evolved_spec
        elif arm == 'routed':
            spec = routed_seed(args.llm_model, policy=json.loads(Path(args.policy).read_text()) if args.policy else None)
        else:
            spec = SEEDS[arm]()
        (arm_dir / 'seed_spec.json').write_text(json.dumps(spec, indent=1))
        name = f'mixedharness_{arm}'
        uses_llm = arm != 'jev_only'
        pool = WorkerPool(size=max(4, 2 * args.concurrency))

        def factory(tools, domain_policy, **kwargs):
            task = kwargs.get('task')
            agent = PipelineAgent(tools, domain_policy, spec, jev, llm if uses_llm else None, label=arm, pool=pool)
            agents[(str(task.id) if task is not None else 'unknown', len(agents))] = agent
            return agent
        registry.register_agent_factory(factory, name)
        agent_name, agent_llm = name, 'none'

    def flush(final=False):
        if not agents:
            return
        snapshot = dict(agents)
        (arm_dir / 'ledgers.json').write_text(json.dumps({f'{k[0]}#{k[1]}': a.ledger for k, a in snapshot.items()}, default=str))
        (arm_dir / 'agents.json').write_text(json.dumps({f'{k[0]}#{k[1]}': a.summary() for k, a in snapshot.items()}, indent=1))
        if not final:
            rows = [row for (task_id, k), a in snapshot.items() for row in a.steps(episode=task_id, arm=arm, trial=k, reward=None)]
            tmp = arm_dir / 'steps.partial.jsonl'
            tmp.write_text(''.join(json.dumps(r, default=str) + '\n' for r in rows))

    stop_flush = threading.Event()

    def flusher():
        while not stop_flush.wait(FLUSH_SECONDS):
            try:
                flush()
            except Exception:
                pass
    if agents is not None and arm != 'llm_only':
        threading.Thread(target=flusher, daemon=True).start()
    config = TextRunConfig(domain=args.domain, task_split_name=adapter.SPLITS[args.split], agent=agent_name,
                           user='replay_user' if args.user == 'replay' else 'user_simulator',
                           llm_agent=agent_llm, llm_args_agent=llm_args_for(agent_llm),
                           llm_user=args.user_llm, llm_args_user=llm_args_for(args.user_llm),
                           num_trials=args.trials, max_steps=args.max_steps, max_concurrency=args.concurrency,
                           seed=args.seed, log_level='ERROR')
    started = time.perf_counter()
    try:
        results = run_tasks(config, tasks, save_path=arm_dir / 'results.json', save_dir=arm_dir, console_display=False)
    finally:
        stop_flush.set()
        if arm != 'llm_only':
            flush(final=True)
            pool.close()
    wall = time.perf_counter() - started
    summary = summarize_results(results, agents if agents else None)
    summary.update({'arm': arm, 'wall_s': wall, 'expected': len(tasks) * args.trials, 'agent_name': agent_name, 'agent_llm': agent_llm})
    if agents:
        steps = _steps_with_rewards(results, agents, arm)
        (arm_dir / 'steps.partial.jsonl').unlink(missing_ok=True)
    else:
        steps = _llm_only_steps(results)
    write_steps(arm_dir / 'steps.jsonl', steps)
    summary['routing'] = route_metrics(steps)
    return summary


def decide(summaries, h):
    s = {x['arm']: x for x in summaries}
    lines = []
    if 'llm_only' in s and s['llm_only']['pass1'] is not None and s['llm_only']['pass1'] < 0.60:
        lines.append('STOP: positive control (llm_only) below 0.60; inspect user simulator and environment before reading anything else.')
    if 'jev_only' in s and (s['jev_only'].get('llm_calls_per_turn') or 0) > 0:
        lines.append('STOP: instrument broken; jev_only made LLM calls.')
    if 'llm_only' in s and (s['llm_only'].get('jev_calls_per_turn') or 0) > 0:
        lines.append('STOP: instrument broken; llm_only made Jev calls.')
    if 'routed' in s and 'llm_only' in s and s['routed']['pass1'] is not None and s['llm_only']['pass1'] is not None and h is not None:
        gap = (s['llm_only']['pass1'] - s['routed']['pass1']) * 100
        best_single = max(v['pass1'] for k, v in s.items() if k in ('llm_only', 'jev_only') and v['pass1'] is not None)
        trail = (best_single - s['routed']['pass1']) * 100
        c_r, c_l = s['routed'].get('agent_cost_per_task_usd'), s['llm_only'].get('agent_cost_per_task_usd')
        ratio = (c_r / c_l) if (c_r is not None and c_l) else None
        if trail > h:
            lines.append(f'FAILS: routed trails the best single arm by {trail:.1f} pt (> ±{h:.1f} noise).')
        elif gap <= h and ratio is not None and ratio <= 0.5:
            lines.append(f'HOLDS: routed within {gap:.1f} pt of llm_only (noise ±{h:.1f}) at {ratio:.2f}x its agent cost.')
        elif gap <= h:
            lines.append(f'INCONCLUSIVE ON COST: routed within {gap:.1f} pt of llm_only but at ' + (f'{ratio:.2f}x' if ratio is not None else 'an unpriced multiple of') + ' its cost (needs <= 0.5x).')
        else:
            lines.append(f'FAILS: routed {gap:.1f} pt below llm_only (> ±{h:.1f} noise).')
    if 'routed_evolved' in s and 'routed' in s and s['routed_evolved']['pass1'] is not None and s['routed']['pass1'] is not None:
        d = (s['routed_evolved']['pass1'] - s['routed']['pass1']) * 100
        lines.append(f'REPORT: routed_evolved is {d:+.1f} pt vs routed (no rule; score it on the held-out half).')
    return lines or ['no decision row fired']


def write_table(summaries, path, resolution_note):
    cols = [('pass^1', 'pass1', '{:.3f}'), ('n', 'completed', '{}'), ('failed', 'failed', '{}'), ('agent $/task', 'agent_cost_per_task_usd', '{:.4f}'),
            ('turn latency median ms', 'turn_latency_ms_median', '{:.0f}'), ('p95 ms', 'turn_latency_ms_p95', '{:.0f}'),
            ('model-only ms', 'turn_model_latency_ms_median', '{:.0f}'), ('sandbox ms', 'turn_sandbox_ms_median', '{:.0f}'),
            ('LLM calls/turn', 'llm_calls_per_turn', '{:.2f}'), ('Jev calls/turn', 'jev_calls_per_turn', '{:.2f}'),
            ('LLM in tok/turn', 'llm_input_tokens_per_turn', '{:.0f}'), ('turns/task', 'turns_per_task', '{:.1f}')]
    route_cols = [('Jev share', 'jev_traffic_share', '{:.2f}'), ('narrow', ('path_mix', 'narrow'), '{:.2f}'), ('escalate', ('path_mix', 'escalate'), '{:.2f}'),
                  ('low conf', ('path_sources', 'low_confidence'), '{:.2f}'), ('fall-through', ('path_sources', 'fallthrough'), '{:.2f}'), ('code override', ('path_sources', 'code_override'), '{:.2f}'),
                  ('guard fallback', ('path_sources', 'guard'), '{:.2f}'), ('agreement', 'agreement_rate', '{:.2f}'), ('guard', 'guard_interventions', '{}')]
    lines = ['| arm | ' + ' | '.join(c[0] for c in cols) + ' |', '|---|' + '---|' * len(cols)]
    for s in summaries:
        cells = []
        for _, key, fmt in cols:
            v = s.get(key)
            cells.append('—' if v is None else fmt.format(v))
        lines.append(f"| {s['arm']} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Routing (from steps.jsonl):', '', '| arm | ' + ' | '.join(c[0] for c in route_cols) + ' |', '|---|' + '---|' * len(route_cols)]
    for s in summaries:
        r = s.get('routing') or {}
        cells = []
        for _, key, fmt in route_cols:
            v = r.get(key[0], {}).get(key[1]) if isinstance(key, tuple) else r.get(key)
            cells.append('—' if v is None else fmt.format(v))
        lines.append(f"| {s['arm']} | " + ' | '.join(cells) + ' |')
    lines += ['', resolution_note]
    path.write_text('\n'.join(lines) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--domain', default='retail')
    p.add_argument('--split', choices=sorted(adapter.SPLITS), default='base')
    p.add_argument('--num-tasks', type=int, default=None)
    p.add_argument('--task-ids', default=None, help='comma-separated tau2 task ids')
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--agent-llm', default='anthropic/claude-sonnet-5', help='litellm model for the llm_only arm')
    p.add_argument('--llm-model', default='claude-sonnet-5', help='Anthropic model for the harness llm and agent nodes')
    p.add_argument('--user-llm', default='anthropic/claude-sonnet-4-5-20250929',
                   help="user simulator (litellm name); tau2's official default is gpt-4.1-2025-04-14, which needs OpenAI credit")
    p.add_argument('--nl-judge-llm', default=None, help='NL-assertion judge (litellm name); defaults to --user-llm')
    p.add_argument('--user', choices=['simulator', 'replay'], default='simulator', help='replay: recorded user turns from --replay-results (plumbing only)')
    p.add_argument('--replay-results', default=None, help='an llm_only results.json whose user turns the replay user says back')
    p.add_argument('--jev-model', default='jev-1.13.0')
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=adapter.DEFAULT_MAX_STEPS)
    p.add_argument('--seed', type=int, default=300)
    p.add_argument('--namespace', default=None, help='response cache namespace for Jev/LLM clients')
    p.add_argument('--evolved-spec', default=None, help='spec JSON for the routed_evolved arm (a candidate from an evolution run)')
    p.add_argument('--policy-steps', default=None, help='steps.jsonl of an earlier run; fits a loop-B policy for routed_evolved when no --evolved-spec is given')
    p.add_argument('--policy', default=None, help='compact policy JSON (scripts/fit_policy.py) embedded in the routed arm: tau per cluster, preferred paths')
    p.add_argument('--seed-module', default='examples.tau2.seeds', help='module holding SEEDS and routed_seed; e.g. examples.tau2.seeds_v3b for the 2026-10-05 harness')
    p.add_argument('--mock', action='store_true', help='mock Jev and LLM for the pipeline arms (plumbing only; scores are meaningless)')
    p.add_argument('--llm-backend', choices=['api', 'codex'], default='api',
                   help='codex: every LLM role (agent, user simulator, judge, harness llm nodes) goes through the ChatGPT subscription; '
                        'tau2 roles via the local proxy at --proxy-base (start scripts/cli_proxy.py first), harness nodes via codex exec directly')
    p.add_argument('--proxy-base', default='http://127.0.0.1:8765/v1')
    p.add_argument('--codex-model', default='gpt-5.5')
    p.add_argument('--out', default='runs/tau2')
    args = p.parse_args(argv)
    _load_seeds(args.seed_module)
    load_dotenv(ROOT / '.env', override=False)
    if args.llm_backend == 'codex':
        args.agent_llm = f'openai/{args.codex_model}'
        args.user_llm = f'openai/{args.codex_model}'
        args.nl_judge_llm = f'openai/{args.codex_model}'
        args.llm_model = args.codex_model
        os.environ['OPENAI_API_BASE'] = args.proxy_base
        os.environ.setdefault('OPENAI_API_KEY', 'subscription')
    arms = [a for a in args.arms.split(',') if a]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        p.error(f'unknown arms {unknown}')
    adapter.ensure_tau2()
    from tau2.runner import get_tasks
    if args.user == 'replay':
        if not args.replay_results:
            p.error('--user replay needs --replay-results')
        from tau2.registry import registry
        from examples.tau2.replay_user import ReplayUser
        ReplayUser.load(args.replay_results)
        registry.register_user(ReplayUser, 'replay_user')
        if not args.task_ids:
            args.task_ids = ','.join(sorted(ReplayUser.recordings))
    nl_judge = args.nl_judge_llm or args.user_llm
    nl_judge_previous = set_nl_judge(nl_judge)
    task_ids = args.task_ids.split(',') if args.task_ids else None
    tasks = get_tasks(args.domain, task_split_name=adapter.SPLITS[args.split], task_ids=task_ids, num_tasks=args.num_tasks)
    if not tasks:
        p.error('no tasks selected')
    if args.user == 'replay':
        from examples.tau2.replay_user import ReplayUser
        ReplayUser.bind(tasks)
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = ROOT / args.out / f'{stamp}-{args.domain}-{args.split}'
    for k in range(1, 100):   # two runs launched in the same second must not share a directory: their flushes would overwrite each other
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            out_dir = ROOT / args.out / f'{stamp}-{k}-{args.domain}-{args.split}'
    namespace = args.namespace or f'tau2-{stamp}'
    if args.mock:
        jev, llm = JevClient(mock=True), LLMClient(kind='mock', model='mock-llm')
    else:
        needs_jev = any(a in PIPELINE_ARMS for a in arms)
        if needs_jev and not os.getenv('TYPESAFE_API_KEY'):
            p.error('TYPESAFE_API_KEY is not set; the jev_only, routed and routed_evolved arms need it')
        jev = JevClient(model=args.jev_model, cache_dir=str(ROOT / '.cache/jev'), cache_namespace=namespace)
        llm = LLMClient(kind='codex' if args.llm_backend == 'codex' else 'anthropic', model=args.llm_model,
                        cache_dir=str(ROOT / '.cache/llm'), cache_namespace=namespace, max_tokens=1024, timeout=180.0)
    evolved_spec, evolved_source = None, None
    if 'routed_evolved' in arms:
        if args.evolved_spec:
            evolved_spec = json.loads(Path(args.evolved_spec).read_text())
            evolved_spec = evolved_spec.get('spec', evolved_spec)
            evolved_source = f'spec {args.evolved_spec}'
        elif args.policy_steps:
            policy = fit_policy(read_steps(args.policy_steps))
            evolved_spec = routed_seed(args.llm_model, policy=policy)
            evolved_source = f'routed seed + loop-B policy fitted on {args.policy_steps}'
        else:
            evolved_spec = routed_seed(args.llm_model)
            evolved_source = 'routed seed unchanged (no --evolved-spec, no --policy-steps): duplicates the routed arm'
    manifest = {'stamp': stamp, 'domain': args.domain, 'split': args.split, 'task_ids': [str(t.id) for t in tasks], 'trials': args.trials,
                'arms': arms, 'agent_llm': args.agent_llm, 'llm_model': args.llm_model, 'user_llm': args.user_llm,
                'nl_judge_llm': nl_judge, 'nl_judge_tau2_default': nl_judge_previous[0],
                'llm_args_agent': llm_args_for(args.agent_llm), 'llm_args_user': llm_args_for(args.user_llm),
                'jev_model': args.jev_model, 'jev_transport': jev.transport, 'mock': args.mock, 'routed_evolved_source': evolved_source,
                'user': args.user, 'replay_results': args.replay_results, 'policy': args.policy, 'seed_module': args.seed_module,
                'llm_backend': args.llm_backend, 'proxy_base': args.proxy_base if args.llm_backend == 'codex' else None, 'cost_imputed': args.llm_backend == 'codex',
                'seed': args.seed, 'max_steps': args.max_steps, 'concurrency': args.concurrency, 'namespace': namespace,
                'git': {'mixed_harness': git_head(ROOT), 'tau2_bench': git_head(Path(os.environ.get('TAU2_HOME') or Path.home() / 'tau2-bench'))},
                'credentials_present': {k: bool(os.getenv(k)) for k in ('TYPESAFE_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY')},
                'started': dt.datetime.now().isoformat()}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    summaries = []
    for arm in arms:
        print(f'== arm {arm}: {len(tasks)} tasks x {args.trials} trials', flush=True)
        summary = run_arm(arm, args, tasks, out_dir, jev, llm, evolved_spec)
        summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k not in ('counters_total',)}, indent=1, default=str), flush=True)
    manifest.update({'finished': dt.datetime.now().isoformat(), 'jev_stats': getattr(jev, 'stats', None), 'llm_stats': getattr(llm, 'stats', None),
                     'jev_actual_models': jev.metadata.get('actual_models')})
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str))
    n = len(tasks) * args.trials
    h = half_width(n)
    note = (f'Resolution: n={n} tasks per arm; 95% half-width at p=0.8 is ±{h:.1f} pt. '
            f'Accuracy gaps below that are reported as within noise. Cost and latency are the discriminating axes.'
            + (' MOCK RUN: scores are meaningless; only the counters are informative.' if args.mock else ''))
    decisions = decide(summaries, h)
    (out_dir / 'summary.json').write_text(json.dumps({'summaries': summaries, 'decisions': decisions, 'resolution_note': note}, indent=1, default=str))
    write_table(summaries, out_dir / 'table.md', note)
    print((out_dir / 'table.md').read_text())
    for line in decisions:
        print('DECISION:', line)
    print(f'artifacts: {out_dir}')


if __name__ == '__main__':
    main()
