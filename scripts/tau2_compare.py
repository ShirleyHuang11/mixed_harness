"""Three arms on tau2-bench, same environment, same user simulator, same tasks and seeds.

Arms
  llm_only   tau2's stock `llm_agent` on --agent-llm (litellm name). One LLM call per turn, no Jev.
  jev_only   examples.tau2.seeds.jev_only_seed(): Jev picks from a code-built catalog, code fills arguments,
             templates reply. No LLM anywhere.
  harness    examples.tau2.seeds.harness_seed(): jev_only plus two llm nodes that fire only when code asks.

Decision table (evaluated automatically after the run; see docs/superpowers/specs/2026-09-24-tau2-three-arm-design.md)
  llm_only pass^1 < 0.60                                   -> STOP: positive control failed; user sim / env suspect
  jev_only llm_calls > 0 or llm_only jev_calls > 0          -> STOP: instrument broken
  harness within 10 pt of llm_only and >= 3x cheaper       -> NEXT: evolution on train (74), score test (40); 3 trials on base
  harness 10-15 pt below llm_only                          -> NEXT: evolution first, then re-score
  harness > 15 pt below after evolution                    -> KILL on tau2; keep instrument; move to settled line

Run inside the tau2-bench virtualenv with this repo installed into it (docs/tau2.md):
  ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --num-tasks 5
  ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --arms jev_only,harness --mock --num-tasks 2   # plumbing only

Outputs runs/tau2/<stamp>/{manifest.json, summary.json, table.md, <arm>/results.json}. table.md is generated
from the artifacts, never typed. Realised counters come from the agents' ledgers and tau2's message ledger.
"""
import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.tau2 import adapter  # noqa: E402
from examples.tau2.seeds import SEEDS  # noqa: E402
from auto_jev.llm import LLMClient  # noqa: E402
from auto_jev.providers import JevClient  # noqa: E402

ARMS = ('llm_only', 'jev_only', 'harness')


NO_TEMPERATURE_MODELS = ('claude-sonnet-5', 'claude-opus-5', 'claude-fable')


def llm_args_for(model):
    """temperature=0 wherever the provider still accepts it; the Claude 5 family rejects the parameter."""
    if any(tag in (model or '') for tag in NO_TEMPERATURE_MODELS):
        return {}
    return {'temperature': 0.0}


def set_nl_judge(model):
    """tau2 reads the NL-assertion judge from a module constant; override it in place and return what was there."""
    import tau2.evaluator.evaluator_nl_assertions as nl
    previous = (nl.DEFAULT_LLM_NL_ASSERTIONS, dict(nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS))
    nl.DEFAULT_LLM_NL_ASSERTIONS = model
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = llm_args_for(model)
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
        r = sim.reward_info.reward
        rewards.append(float(r))
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


def run_arm(arm, args, tasks, out_dir, jev, llm):
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
        spec = SEEDS[arm](args.llm_model) if arm == 'harness' else SEEDS[arm]()
        (arm_dir / 'seed_spec.json').write_text(json.dumps(spec, indent=1))
        name = f'auto_jev_{arm}'

        def factory(tools, domain_policy, **kwargs):
            task = kwargs.get('task')
            agent = PipelineAgent(tools, domain_policy, spec, jev, llm if arm == 'harness' else None, label=arm)
            agents[(str(task.id) if task is not None else 'unknown', len(agents))] = agent
            return agent
        registry.register_agent_factory(factory, name)
        agent_name, agent_llm = name, 'none'
    config = TextRunConfig(domain=args.domain, task_split_name=adapter.SPLITS[args.split], agent=agent_name,
                           llm_agent=agent_llm, llm_args_agent=llm_args_for(agent_llm),
                           llm_user=args.user_llm, llm_args_user=llm_args_for(args.user_llm),
                           num_trials=args.trials, max_steps=args.max_steps, max_concurrency=args.concurrency,
                           seed=args.seed, log_level='ERROR')
    started = time.perf_counter()
    results = run_tasks(config, tasks, save_path=arm_dir / 'results.json', save_dir=arm_dir, console_display=False)
    wall = time.perf_counter() - started
    summary = summarize_results(results, agents if agents else None)
    summary.update({'arm': arm, 'wall_s': wall, 'expected': len(tasks) * args.trials, 'agent_name': agent_name, 'agent_llm': agent_llm})
    if agents:
        ledgers = {f'{k[0]}#{k[1]}': a.ledger for k, a in agents.items()}
        (arm_dir / 'ledgers.json').write_text(json.dumps(ledgers, default=str))
        (arm_dir / 'agents.json').write_text(json.dumps({f'{k[0]}#{k[1]}': a.summary() for k, a in agents.items()}, indent=1))
    return summary


def decide(summaries):
    s = {x['arm']: x for x in summaries}
    lines = []
    if 'llm_only' in s and s['llm_only']['pass1'] is not None and s['llm_only']['pass1'] < 0.60:
        lines.append('STOP: positive control (llm_only) below 0.60; inspect user simulator and environment before reading anything else.')
    if 'jev_only' in s and s['jev_only'].get('llm_calls_per_turn', 0) > 0:
        lines.append('STOP: instrument broken; jev_only made LLM calls.')
    if 'llm_only' in s and s['llm_only'].get('jev_calls_per_turn', 0) > 0:
        lines.append('STOP: instrument broken; llm_only made Jev calls.')
    if 'harness' in s and 'llm_only' in s and s['harness']['pass1'] is not None and s['llm_only']['pass1'] is not None:
        gap = (s['llm_only']['pass1'] - s['harness']['pass1']) * 100
        c_h, c_l = s['harness'].get('agent_cost_per_task_usd'), s['llm_only'].get('agent_cost_per_task_usd')
        cheaper = (c_l / c_h) if (c_h and c_l) else None
        if gap <= 10 and (cheaper is None or cheaper >= 3):
            lines.append(f'NEXT: harness within {gap:.1f} pt of llm_only' + (f' and {cheaper:.1f}x cheaper' if cheaper else ' (cost unpriced)') + '; run evolution on train, score test, then 3 trials on base.')
        elif gap <= 15:
            lines.append(f'NEXT: harness {gap:.1f} pt below llm_only; run evolution first, then re-score.')
        else:
            lines.append(f'KILL-CANDIDATE: harness {gap:.1f} pt below llm_only; confirm after evolution, then move the instrument to the settled line.')
    if 'harness' in s and 'jev_only' in s and s['jev_only']['pass1'] and s['harness']['pass1'] is not None:
        ratio = s['harness']['pass1'] / s['jev_only']['pass1'] if s['jev_only']['pass1'] else None
        if ratio is not None and ratio < 1.5:
            lines.append(f'WARN: harness/jev_only pass^1 ratio {ratio:.2f} < 1.5; check llm_calls_per_turn before interpreting.')
    return lines or ['no decision row fired']


def write_table(summaries, path, resolution_note):
    cols = [('pass^1', 'pass1', '{:.3f}'), ('n', 'completed', '{}'), ('failed', 'failed', '{}'), ('agent $/task', 'agent_cost_per_task_usd', '{:.4f}'),
            ('turn latency median ms', 'turn_latency_ms_median', '{:.0f}'), ('p95 ms', 'turn_latency_ms_p95', '{:.0f}'),
            ('model-only ms', 'turn_model_latency_ms_median', '{:.0f}'), ('sandbox ms', 'turn_sandbox_ms_median', '{:.0f}'),
            ('LLM calls/turn', 'llm_calls_per_turn', '{:.2f}'), ('Jev calls/turn', 'jev_calls_per_turn', '{:.2f}'),
            ('LLM in tok/turn', 'llm_input_tokens_per_turn', '{:.0f}'), ('turns/task', 'turns_per_task', '{:.1f}')]
    lines = ['| arm | ' + ' | '.join(c[0] for c in cols) + ' |', '|---|' + '---|' * len(cols)]
    for s in summaries:
        cells = []
        for _, key, fmt in cols:
            v = s.get(key)
            cells.append('—' if v is None else fmt.format(v))
        lines.append(f"| {s['arm']} | " + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append(resolution_note)
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
    p.add_argument('--llm-model', default='claude-sonnet-5', help='Anthropic model for the harness llm nodes')
    p.add_argument('--user-llm', default='anthropic/claude-sonnet-4-5-20250929',
                   help="user simulator (litellm name); tau2's official default is gpt-4.1-2025-04-14, which needs OpenAI credit")
    p.add_argument('--nl-judge-llm', default=None, help='NL-assertion judge (litellm name); defaults to --user-llm')
    p.add_argument('--jev-transport', choices=['auto', 'typesafe', 'vercel'], default='auto')
    p.add_argument('--jev-model', default='typesafe-ai/jev')
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=adapter.DEFAULT_MAX_STEPS)
    p.add_argument('--seed', type=int, default=300)
    p.add_argument('--namespace', default=None, help='response cache namespace for Jev/LLM clients')
    p.add_argument('--mock', action='store_true', help='mock Jev and LLM for the pipeline arms (plumbing only; scores are meaningless)')
    p.add_argument('--out', default='runs/tau2')
    args = p.parse_args(argv)
    load_dotenv(ROOT / '.env', override=False)
    arms = [a for a in args.arms.split(',') if a]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        p.error(f'unknown arms {unknown}')
    adapter.ensure_tau2()
    from tau2.runner import get_tasks
    nl_judge = args.nl_judge_llm or args.user_llm
    nl_judge_previous = set_nl_judge(nl_judge)
    task_ids = args.task_ids.split(',') if args.task_ids else None
    tasks = get_tasks(args.domain, task_split_name=adapter.SPLITS[args.split], task_ids=task_ids, num_tasks=args.num_tasks)
    if not tasks:
        p.error('no tasks selected')
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = ROOT / args.out / f'{stamp}-{args.domain}-{args.split}'
    out_dir.mkdir(parents=True, exist_ok=True)
    namespace = args.namespace or f'tau2-{stamp}'
    if args.mock:
        jev, llm = JevClient(mock=True), LLMClient(kind='mock', model='mock-llm')
    else:
        transport = args.jev_transport if args.jev_transport != 'auto' else ('vercel' if os.getenv('AI_GATEWAY_API_KEY') else 'typesafe')
        needs_jev = any(a in ('jev_only', 'harness') for a in arms)
        key_env = 'AI_GATEWAY_API_KEY' if transport == 'vercel' else 'TYPESAFE_API_KEY'
        if needs_jev and not os.getenv(key_env):
            p.error(f'{key_env} is not set; the jev_only and harness arms need it')
        jev = JevClient(transport=transport, model=args.jev_model, cache_dir=str(ROOT / '.cache/jev'), cache_namespace=namespace)
        llm = LLMClient(kind='anthropic', model=args.llm_model, cache_dir=str(ROOT / '.cache/llm'), cache_namespace=namespace, max_tokens=800)
    manifest = {'stamp': stamp, 'domain': args.domain, 'split': args.split, 'task_ids': [str(t.id) for t in tasks], 'trials': args.trials,
                'arms': arms, 'agent_llm': args.agent_llm, 'llm_model': args.llm_model, 'user_llm': args.user_llm,
                'nl_judge_llm': nl_judge, 'nl_judge_tau2_default': nl_judge_previous[0],
                'llm_args_agent': llm_args_for(args.agent_llm), 'llm_args_user': llm_args_for(args.user_llm),
                'jev_model': args.jev_model, 'jev_transport': getattr(jev, 'transport', 'mock'), 'mock': args.mock,
                'seed': args.seed, 'max_steps': args.max_steps, 'concurrency': args.concurrency, 'namespace': namespace,
                'git': {'mixed_harness': git_head(ROOT), 'tau2_bench': git_head(Path(os.environ.get('TAU2_HOME') or Path.home() / 'tau2-bench'))},
                'credentials_present': {k: bool(os.getenv(k)) for k in ('TYPESAFE_API_KEY', 'AI_GATEWAY_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY')},
                'started': dt.datetime.now().isoformat()}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    summaries = []
    for arm in arms:
        print(f'== arm {arm}: {len(tasks)} tasks x {args.trials} trials', flush=True)
        summary = run_arm(arm, args, tasks, out_dir, jev, llm)
        summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k not in ('counters_total',)}, indent=1, default=str), flush=True)
    manifest.update({'finished': dt.datetime.now().isoformat(), 'jev_stats': getattr(jev, 'stats', None), 'llm_stats': getattr(llm, 'stats', None),
                     'jev_actual_model': (getattr(jev, 'metadata', {}) or {}).get('actual_model')})
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str))
    n = len(tasks) * args.trials
    pbar = 0.8
    half_width = 1.96 * (pbar * (1 - pbar) / n) ** 0.5 * 100 if n else None
    note = (f'Resolution: n={n} tasks per arm; 95% half-width at p=0.8 is ±{half_width:.1f} pt. '
            f'Accuracy gaps below that are reported as within noise. Cost and latency are the discriminating axes.'
            + (' MOCK RUN: scores are meaningless; only the counters are informative.' if args.mock else ''))
    decisions = decide(summaries)
    (out_dir / 'summary.json').write_text(json.dumps({'summaries': summaries, 'decisions': decisions, 'resolution_note': note}, indent=1, default=str))
    write_table(summaries, out_dir / 'table.md', note)
    print((out_dir / 'table.md').read_text())
    for line in decisions:
        print('DECISION:', line)
    print(f'artifacts: {out_dir}')


if __name__ == '__main__':
    main()
