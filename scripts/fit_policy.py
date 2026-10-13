"""Loop B offline: fit the per-cluster threshold and path preference, write the compact policy a seed embeds.

    ~/tau2-bench/.venv/bin/python scripts/fit_policy.py --replay runs/stage0/<stamp>/replay.jsonl \
        --steps runs/tau2/<run>/routed/steps.jsonl --out examples/tau2/policy.json

The threshold per cluster comes from the stage-0 replay (Jev's pick against the LLM's action on recorded LLM-only
turns, a per-step signal): the smallest tau on a grid at which agreement at or above tau reaches --target. The path
preference per cluster comes from routed step tables joined to episode rewards (fit_policy). Both land in one JSON
the runner passes to `routed_seed(policy=...)` through `--policy`, and the file is part of the harness version.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixedharness.router import compact_policy, fit_from_steps, fit_policy, fit_tau, merge_policy  # noqa: E402
from scripts.learn_splits import fit_splits, load_steps  # noqa: E402
from mixedharness.store import read_jsonl  # noqa: E402


def replay_rows_from_ledgers(paths):
    """Replay rows from recorded routed ledgers: on every step the LLM took, Jev had already picked (the decision node's
    action inside the recorded runtime result) with a probability, so the pair (Jev pick, LLM action) is an agreement observation like a stage-0 replay row.
    Steps Jev took are not used: the LLM's action there is unknown. Tool picks agree on the tool name; a reply agrees
    with a reply."""
    rows = []
    for path in paths:
        for key, entries in json.loads(Path(path).read_text()).items():
            for e in entries:
                r = e.get('routing') or {}
                if r.get('path_source') not in ('low_confidence', 'escalate_below_tau') or r.get('confidence') is None:
                    continue
                nodes = (e.get('decision') or {}).get('nodes') or {}   # the ledger's decision record is the whole runtime result
                jev = (nodes.get('decision') or {}).get('action') or {}   # the closed-path action built from Jev's own pick
                llm = e.get('action') or {}
                if not jev or not llm:
                    continue
                agree = (jev.get('kind') == llm.get('kind')) and (jev.get('kind') != 'tool' or jev.get('name') == llm.get('name'))
                rows.append({'task_id': key.split('#')[0], 'cluster': r.get('cluster'), 'label_closed': True, 'jev_confidence': float(r['confidence']),
                             'jev_correct': bool(agree), 'source': 'ledger'})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--replay', nargs='*', default=[], help='stage-0 replay.jsonl files')
    p.add_argument('--ledger-replay', nargs='*', default=[], help='routed ledgers.json files: the steps the LLM took become replay rows (Jev pick vs LLM action) and refit the thresholds')
    p.add_argument('--tau-only', action='store_true', help='fit only the thresholds (the 2026-10-05 table); no code rows, no split rows')
    p.add_argument('--steps', nargs='*', default=[], help='routed steps.jsonl files with episode rewards')
    p.add_argument('--ledgers', nargs='*', default=[], help='routed ledgers.json files: split rows (cluster + state features -> action) are learned from the LLM steps in them')
    p.add_argument('--target', type=float, default=0.9)
    p.add_argument('--min-n', type=int, default=10)
    p.add_argument('--min-samples', type=int, default=8)
    p.add_argument('--epsilon', type=float, default=0.0)
    p.add_argument('--judge-json', default=None, help='judge.json from scripts/calibrate_judge.py: the judge spec and temperature this table was fitted for, embedded as policy["judge"]')
    p.add_argument('--out', default='examples/tau2/policy.json')
    args = p.parse_args(argv)
    replay = [r for f in args.replay for r in read_jsonl(f)]
    replay += replay_rows_from_ledgers(args.ledger_replay)
    steps = [r for f in args.steps for r in read_jsonl(f)]
    tau = fit_tau(replay, target=args.target, min_n=args.min_n) if replay else None
    paths = fit_policy(steps, min_samples=args.min_samples, epsilon=args.epsilon) if steps else None
    learned = fit_from_steps(steps, min_n=args.min_n, replay=replay) if steps and not args.tau_only else None
    splits = fit_splits(load_steps(args.ledgers), min_n=args.min_n) if args.ledgers and not args.tau_only else None
    full = merge_policy(paths, tau, epsilon=args.epsilon, learned=learned, splits=splits)
    compact = compact_policy(full)
    if args.judge_json:
        cal = json.loads(Path(args.judge_json).read_text())
        compact['judge'] = {'spec': cal['judge'], 'temperature': float(cal.get('temperature') or 1.0)}
    elif replay and all(r.get('judge') for r in replay[:5]) and len({r.get('judge') for r in replay}) == 1:
        compact['judge'] = {'spec': replay[0]['judge'], 'temperature': 1.0}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**compact, 'provenance': {'replay': args.replay, 'steps': args.steps, 'target': args.target, 'min_n': args.min_n,
                                                        'min_samples': args.min_samples, 'replay_rows': len(replay), 'step_rows': len(steps)},
                               'tau_detail': (tau or {}).get('clusters', {}), 'path_detail': {c: v.get('stats') for c, v in ((paths or {}).get('clusters') or {}).items()},
                               'learned_detail': (learned or {}).get('stats', {})},
                              indent=1, sort_keys=True))
    print('| cluster | closed turns | tau | accepted | agreement at tau | preferred path |')
    print('|---|---|---|---|---|---|')
    clusters = sorted(set((tau or {}).get('clusters', {})) | set(compact['prefer']), key=lambda c: -((tau or {}).get('clusters', {}).get(c, {}).get('n', 0)))
    for c in clusters:
        t = (tau or {}).get('clusters', {}).get(c, {})
        print(f"| {c} | {t.get('n', '—')} | {t.get('tau', 'default')} | {t.get('accepted', '—')} | {('%.2f' % t['agreement']) if t.get('agreement') is not None else '—'} | {compact['prefer'].get(c, '—')} |")
    if splits:
        print(f"split rows learned from ledgers: {len(splits['rows'])}, covering {splits['covered']} of {splits['n_steps']} LLM steps")
    if learned:
        print('code rows learned from steps:', compact['defaults'])
        print('tightened to never:', learned['tighten'])
    print(f'written: {out}')


if __name__ == '__main__':
    main()
