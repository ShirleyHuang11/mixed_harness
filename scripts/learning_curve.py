"""The learning curve: run -> recompute the table -> run, for several rounds, on the same tasks.

    ~/tau2-bench/.venv/bin/python scripts/learning_curve.py --rounds 3 --llm-backend codex \
        --replay runs/stage0/<stamp>/replay.jsonl --seed-steps runs/tau2/<run>/routed/steps.jsonl --out runs/curve/<name>

Round r fits the table from the stage-0 replay plus every routed step table accumulated so far (the seed steps and
rounds 0..r-1), writes it to <out>/policy_r.json, runs the routed arm on the tasks with that table, and appends one
line to <out>/curve.jsonl: pass^1, agent cost per task, turns per task, the share of steps taken by code, Jev and
the LLM, and the number of code rows and never-rows in the table. Round 0 uses the table fitted from the seed steps
alone. Every tau2_compare run keeps its own directory under runs/tau2/ as usual; this script only orchestrates and
reads their summary.json and steps.jsonl.

What "learning" means here, concretely: a state type moves to code when the harness's own accepted picks there were
the same option >= 95% of the time in won episodes; a state type's threshold tightens to never when its accepted
Jev steps sit in lost episodes at > 1.5x the overall rate; thresholds themselves come from the replay and do not
change between rounds. The curve is the test of "traffic moves toward the cheaper end while accuracy holds".
"""
import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixedharness.store import append_jsonl, read_jsonl  # noqa: E402


def run(cmd, log):
    """Run a command, append its output to the log, and return (returncode, the `artifacts:` directory it printed or None)."""
    with open(log, 'a') as f:
        f.write('\n$ ' + ' '.join(cmd) + '\n')
        f.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(proc.stdout)
    art = [line.split(':', 1)[1].strip() for line in proc.stdout.splitlines() if line.startswith('artifacts:')]
    return proc.returncode, (Path(art[-1]) if art else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--replay', required=True, help='stage-0 replay.jsonl (thresholds)')
    p.add_argument('--seed-steps', nargs='*', default=[], help='routed steps.jsonl from earlier runs, used for round 0')
    p.add_argument('--out', default=None)
    p.add_argument('--llm-backend', choices=['api', 'codex'], default='codex')
    p.add_argument('--num-tasks', type=int, default=None)
    p.add_argument('--task-ids', default=None)
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--judge', default=None, help='judge behind jev nodes (jev, jev:<model>, logprob:<model>@<base_url>); --replay must come from the same judge')
    args = p.parse_args(argv)
    out = Path(args.out) if args.out else ROOT / 'runs/curve' / dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'curve.log'
    python = sys.executable
    steps_files = list(args.seed_steps)
    ledger_files = [str(Path(f).with_name('ledgers.json')) for f in steps_files if Path(f).with_name('ledgers.json').exists()]
    for r in range(args.rounds):
        policy = out / f'policy_{r}.json'
        cmd = [python, 'scripts/fit_policy.py', '--replay', args.replay, '--out', str(policy)]
        if steps_files:
            cmd += ['--steps'] + steps_files
        if ledger_files:
            cmd += ['--ledgers'] + ledger_files
        if run(cmd, log)[0]:
            raise SystemExit(f'fit failed in round {r}; see {log}')
        pol = json.loads(policy.read_text())
        cmd = [python, 'scripts/tau2_compare.py', '--domain', 'retail', '--split', 'base', '--arms', 'routed', '--policy', str(policy),
               '--concurrency', str(args.concurrency), '--llm-backend', args.llm_backend]
        if args.num_tasks:
            cmd += ['--num-tasks', str(args.num_tasks)]
        if args.task_ids:
            cmd += ['--task-ids', args.task_ids]
        if args.judge:
            cmd += ['--judge', args.judge]
        rc, run_dir = run(cmd, log)   # the run's own `artifacts:` line, not the newest directory: other runs may share the clock
        if rc or run_dir is None:
            raise SystemExit(f'run failed in round {r}; see {log}')
        summary = json.loads((run_dir / 'summary.json').read_text())['summaries'][0]
        routing = summary.get('routing') or {}
        sources = routing.get('path_sources') or {}
        row = {'round': r, 'run_dir': str(run_dir), 'policy': str(policy), 'pass1': summary.get('pass1'), 'completed': summary.get('completed'),
               'failed': summary.get('failed'), 'agent_cost_per_task_usd': summary.get('agent_cost_per_task_usd'),
               'turns_per_task': summary.get('turns_per_task'), 'llm_calls_per_turn': summary.get('llm_calls_per_turn'),
               'jev_calls_per_turn': summary.get('jev_calls_per_turn'),
               'share_code': sources.get('structural', 0.0) + sources.get('learned', 0.0), 'share_jev': sources.get('jev', 0.0),
               'escalate_share': (routing.get('path_mix') or {}).get('escalate'),
               'code_rows': len(pol.get('defaults') or {}), 'split_rows': len(pol.get('splits') or []), 'never_rows': sum(1 for v in (pol.get('tau') or {}).values() if v > 1.0),
               'at': dt.datetime.now().isoformat()}
        append_jsonl(out / 'curve.jsonl', row)
        print(json.dumps(row), flush=True)
        steps_files.append(str(run_dir / 'routed' / 'steps.jsonl'))
        ledger_files.append(str(run_dir / 'routed' / 'ledgers.json'))
    rows = read_jsonl(out / 'curve.jsonl')
    print('| round | pass^1 | $/task | turns | code | Jev | escalate | code rows | split rows | never rows |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for x in rows:
        print(f"| {x['round']} | {x['pass1']:.3f} | {x['agent_cost_per_task_usd'] or 0:.3f} | {x['turns_per_task']:.1f} | {x['share_code']:.2f} | {x['share_jev']:.2f} | {x['escalate_share']:.2f} | {x['code_rows']} | {x.get('split_rows', 0)} | {x['never_rows']} |")
    print(f'artifacts: {out}')


if __name__ == '__main__':
    main()
