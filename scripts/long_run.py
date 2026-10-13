"""The long run: a stream of conversations the table learns from, scored on conversations it never saw.

    ~/tau2-bench/.venv/bin/python scripts/long_run.py --passes 6 --replay runs/stage0/<stamp>/replay.jsonl \
        --llm-only-run runs/tau2/<llm_only run> --out runs/longrun/<name>

Why this shape (chosen over re-running all 114 tasks round after round): a round-over-round curve on the same
tasks cannot separate learning from memorisation, and a deployed harness is judged on the conversations it has
not seen. So the 114 retail tasks are split once (seeded) into a STREAM of 76 tasks and a HELD-OUT set of 38.
Each pass p: fit the table from the stage-0 replay restricted to stream tasks plus every stream pass so far
(their step tables and ledgers; held-out runs never enter a fit), then run the stream pass with a new simulator
seed and, in parallel, the held-out set with the same table and a fixed seed. One row per pass: cumulative
conversations served, stream and held-out pass^1, imputed cost per conversation, code/Jev/LLM shares, rows in
the table. The llm_only reference is the existing 114-task llm_only run under the same simulator (its per-task
cost and reward give the ratio on each subset); its noise on 38 tasks is wide, so the held-out accuracy test is
"within the noise band", not "better".

Prediction written before the first run (2026-10-08): by pass 6 the held-out LLM share falls from ~0.65 to <= 0.45
and cost per conversation from ~0.73x to <= 0.55x of llm_only on the same 38 tasks, with held-out pass^1 within
9 points of llm_only's. If the held-out shares do not move while the stream's do, the rows are task-specific and
the claim has to be narrowed to "within a task mix".

Three candidate shapes were ranked by upside before choosing this one: (A) this stream + held-out design on
retail, startable today; (B) a stream that interleaves retail and airline, higher upside (one table serving two
domains) but airline has no option-listing compiler yet; (C) six more rounds on all 114 tasks, cheapest, rejected
because it measures memorisation. B is the upgrade once an airline compiler exists.
"""
import argparse
import datetime as dt
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixedharness.store import append_jsonl, read_jsonl  # noqa: E402

PY = sys.executable


def run(cmd, log):
    with open(log, 'a') as f:
        f.write('\n$ ' + ' '.join(cmd) + '\n')
        f.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(proc.stdout)
    art = [line.split(':', 1)[1].strip() for line in proc.stdout.splitlines() if line.startswith('artifacts:')]
    return proc.returncode, (Path(art[-1]) if art else None)


def start(cmd, log):
    f = open(log, 'a')
    f.write('\n$ ' + ' '.join(cmd) + '\n')
    f.flush()
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True), f


def finish(proc, f, log):
    proc.wait()
    f.close()
    text = Path(log).read_text()
    art = [line.split(':', 1)[1].strip() for line in text.splitlines() if line.startswith('artifacts:')]
    return proc.returncode, (Path(art[-1]) if art else None)


def per_task(run_dir, arm):
    """task id -> (reward, imputed agent cost) from a run's results and ledgers (llm_only: tau2's agent_cost)."""
    r = json.loads((run_dir / arm / 'results.json').read_text())
    out = {}
    for s in r['simulations']:
        if s.get('reward_info') and s['reward_info'].get('reward') is not None:
            out[str(s['task_id'])] = [float(s['reward_info']['reward']), float(s.get('agent_cost') or 0.0)]
    led = run_dir / arm / 'ledgers.json'
    if led.exists():
        for key, entries in json.loads(led.read_text()).items():
            t = key.split('#')[0]
            if t in out:
                out[t][1] = sum(e.get('cost_usd', 0.0) for e in entries)
    return out


def shares(run_dir, arm):
    from collections import Counter
    c = Counter()
    led = run_dir / arm / 'ledgers.json'
    if not led.exists():
        return {}
    for key, entries in json.loads(led.read_text()).items():
        for e in entries:
            c[(e.get('routing') or {}).get('path_source')] += 1
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def block(pt, ids):
    rows = [pt[t] for t in ids if t in pt]
    if not rows:
        return {'n': 0, 'pass1': None, 'cost': None}
    return {'n': len(rows), 'pass1': sum(r[0] for r in rows) / len(rows), 'cost': sum(r[1] for r in rows) / len(rows)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--passes', type=int, default=6)
    p.add_argument('--replay', required=True)
    p.add_argument('--llm-only-run', required=True, help='an llm_only run on all 114 tasks under the same simulator')
    p.add_argument('--out', default=None)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--n-holdout', type=int, default=38)
    p.add_argument('--n-tasks', type=int, default=114)
    p.add_argument('--stream-limit', type=int, default=None, help='smoke: use only this many stream tasks')
    p.add_argument('--holdout-limit', type=int, default=None, help='smoke: use only this many held-out tasks')
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--llm-backend', choices=['api', 'codex'], default='codex')
    p.add_argument('--proxy-stream', default='http://127.0.0.1:8766/v1')
    p.add_argument('--proxy-holdout', default='http://127.0.0.1:8767/v1')
    p.add_argument('--seed-module', default='examples.tau2.seeds', help='harness version to run (examples.tau2.seeds_v3b = the 2026-10-05 figure)')
    p.add_argument('--tau-only', action='store_true', help='the table learns only thresholds, refit each pass from the replay plus the stream ledgers (the 2026-10-05 evolution loop)')
    p.add_argument('--first-pass', type=int, default=0, help='number the first pass this way (resume)')
    p.add_argument('--judge', default=None, help='judge behind jev nodes (jev, jev:<model>, logprob:<model>@<base_url>); --replay must come from the same judge')
    p.add_argument('--judge-temperature', type=float, default=None)
    p.add_argument('--seed-stream-dirs', nargs='*', default=[], help='stream run directories of earlier passes: their steps and ledgers enter every fit (resume)')
    args = p.parse_args(argv)
    out = Path(args.out) if args.out else ROOT / 'runs/longrun' / dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'longrun.log'
    ids = [str(i) for i in range(args.n_tasks)]
    rng = random.Random(args.split_seed)
    rng.shuffle(ids)
    holdout = sorted(ids[:args.n_holdout], key=int)
    stream = sorted(ids[args.n_holdout:], key=int)
    if args.holdout_limit:
        holdout = holdout[:args.holdout_limit]
    if args.stream_limit:
        stream = stream[:args.stream_limit]
    (out / 'split.json').write_text(json.dumps({'stream': stream, 'holdout': holdout, 'split_seed': args.split_seed}, indent=1))
    # the replay restricted to stream tasks: thresholds never see a held-out conversation
    replay_stream = out / 'replay_stream.jsonl'
    with open(replay_stream, 'w') as f:
        for row in read_jsonl(args.replay):
            if str(row.get('task_id')) in set(stream):
                f.write(json.dumps(row) + '\n')
    ref = per_task(Path(args.llm_only_run), 'llm_only')
    ref_stream, ref_holdout = block(ref, stream), block(ref, holdout)
    print('llm_only reference: stream', ref_stream, '| holdout', ref_holdout, flush=True)
    steps_files = [str(Path(d) / 'routed' / 'steps.jsonl') for d in args.seed_stream_dirs]
    ledger_files = [str(Path(d) / 'routed' / 'ledgers.json') for d in args.seed_stream_dirs]
    served = len(stream) * len(args.seed_stream_dirs)
    for q in range(args.first_pass, args.first_pass + args.passes):
        policy = out / f'policy_{q}.json'
        cmd = [PY, 'scripts/fit_policy.py', '--replay', str(replay_stream), '--out', str(policy)]
        if args.tau_only:
            cmd += ['--tau-only']
            if ledger_files:
                cmd += ['--ledger-replay'] + ledger_files
        else:
            if steps_files:
                cmd += ['--steps'] + steps_files
            if ledger_files:
                cmd += ['--ledgers'] + ledger_files
        if run(cmd, log)[0]:
            raise SystemExit(f'fit failed in pass {q}; see {log}')
        pol = json.loads(policy.read_text())
        base = [PY, 'scripts/tau2_compare.py', '--domain', 'retail', '--split', 'base', '--arms', 'routed', '--policy', str(policy),
                '--concurrency', str(args.concurrency), '--llm-backend', args.llm_backend, '--seed-module', args.seed_module]
        if args.judge:
            base += ['--judge', args.judge]
        if args.judge_temperature is not None:
            base += ['--judge-temperature', str(args.judge_temperature)]
        s_log, h_log = out / f'stream_{q}.log', out / f'holdout_{q}.log'
        sp, sf = start(base + ['--task-ids', ','.join(stream), '--seed', str(300 + q), '--proxy-base', args.proxy_stream], s_log)
        hp, hf = start(base + ['--task-ids', ','.join(holdout), '--seed', '900', '--proxy-base', args.proxy_holdout], h_log)
        s_rc, s_dir = finish(sp, sf, s_log)
        h_rc, h_dir = finish(hp, hf, h_log)
        if s_rc or s_dir is None or h_rc or h_dir is None:
            raise SystemExit(f'pass {q} failed (stream rc {s_rc}, holdout rc {h_rc}); see {out}')
        served += len(stream)
        s_pt, h_pt = per_task(s_dir, 'routed'), per_task(h_dir, 'routed')
        s_blk, h_blk = block(s_pt, stream), block(h_pt, holdout)
        s_sh, h_sh = shares(s_dir, 'routed'), shares(h_dir, 'routed')
        row = {'pass': q, 'served': served, 'policy': str(policy), 'stream_dir': str(s_dir), 'holdout_dir': str(h_dir),
               'stream': s_blk, 'holdout': h_blk,
               'stream_cost_ratio': (s_blk['cost'] / ref_stream['cost']) if s_blk['cost'] and ref_stream['cost'] else None,
               'holdout_cost_ratio': (h_blk['cost'] / ref_holdout['cost']) if h_blk['cost'] and ref_holdout['cost'] else None,
               'holdout_pass_gap': (h_blk['pass1'] - ref_holdout['pass1']) if h_blk['pass1'] is not None and ref_holdout['pass1'] is not None else None,
               'stream_shares': {k: round(v, 3) for k, v in s_sh.items()}, 'holdout_shares': {k: round(v, 3) for k, v in h_sh.items()},
               'seed_module': args.seed_module, 'tau_rows': len(pol.get('tau') or {}), 'code_rows': len(pol.get('defaults') or {}), 'split_rows': len(pol.get('splits') or []),
               'never_rows': sum(1 for v in (pol.get('tau') or {}).values() if v > 1.0), 'at': dt.datetime.now().isoformat()}
        append_jsonl(out / 'curve.jsonl', row)
        print(json.dumps(row), flush=True)
        steps_files.append(str(s_dir / 'routed' / 'steps.jsonl'))
        ledger_files.append(str(s_dir / 'routed' / 'ledgers.json'))
    rows = read_jsonl(out / 'curve.jsonl')
    print(f"llm_only reference: stream pass {ref_stream['pass1']:.3f} ${ref_stream['cost']:.3f} | holdout pass {ref_holdout['pass1']:.3f} ${ref_holdout['cost']:.3f}")
    print('| pass | served | stream pass^1 | stream $ | x | held-out pass^1 | held-out $ | x | held-out LLM share | split rows |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for x in rows:
        hs = x['holdout_shares']
        llm = sum(v for k, v in hs.items() if k in ('low_confidence', 'code_override', 'trajectory', 'forced'))
        print(f"| {x['pass']} | {x['served']} | {x['stream']['pass1']:.3f} | {x['stream']['cost']:.3f} | {x['stream_cost_ratio'] or 0:.2f} | {x['holdout']['pass1']:.3f} | {x['holdout']['cost']:.3f} | {x['holdout_cost_ratio'] or 0:.2f} | {llm:.2f} | {x['split_rows']} |")
    print(f'artifacts: {out}')


if __name__ == '__main__':
    main()
