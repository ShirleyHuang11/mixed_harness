"""Merge several tau2_compare run directories into one table: union of simulations per arm, deduplicated by task.

    ~/tau2-bench/.venv/bin/python scripts/tau2_merge.py runs/tau2/<a> runs/tau2/<b> --out runs/tau2/<merged>

For each arm found in any input directory, the simulations of all inputs are collected; when a task appears more
than once the completed one wins (a rewarded simulation over an infrastructure error; later directories over earlier
ones on ties). Ledgers, agents and steps are merged the same way, keyed by task. The merged directory gets a fresh
`summary.json` and `table.md` computed by the same functions the runner uses, plus a `merged_from` list in its
manifest. Arms are compared on the intersection of their task ids as well, reported in `summary.json` under
`common_tasks`, because a pass^1 over different task sets is not a comparison.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.tau2 import adapter  # noqa: E402
from mixedharness.router import route_metrics  # noqa: E402
from mixedharness.store import read_jsonl  # noqa: E402


def _completed(sim):
    return sim.get('reward_info') is not None and sim['reward_info'].get('reward') is not None


def merge_results(paths):
    """List of results.json paths (same arm) -> merged Results dict, keyed by task id, completed wins."""
    merged, head = {}, None
    for path in paths:
        data = json.loads(Path(path).read_text())
        head = head or {k: v for k, v in data.items() if k != 'simulations'}
        for sim in data.get('simulations') or []:
            key = str(sim.get('task_id'))
            if key not in merged or (_completed(sim) and not _completed(merged[key])) or (_completed(sim) == _completed(merged[key])):
                merged[key] = sim
    head = head or {}
    head['simulations'] = [merged[k] for k in sorted(merged, key=lambda t: (len(t), t))]
    if 'tasks' in head:
        head['tasks'] = [t for t in head['tasks'] if str(t.get('id')) in merged] if isinstance(head['tasks'], list) else head['tasks']
    return head


def merge_keyed_json(paths):
    out = {}
    for path in paths:
        if Path(path).is_file():
            out.update(json.loads(Path(path).read_text()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dirs', nargs='+')
    p.add_argument('--out', default=None)
    args = p.parse_args(argv)
    adapter.ensure_tau2()
    from tau2.data_model.simulation import Results
    from scripts.tau2_compare import decide, half_width, summarize_results, write_table
    dirs = [Path(d) for d in args.dirs]
    out = Path(args.out) if args.out else ROOT / 'runs/tau2' / (dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-merged')
    out.mkdir(parents=True, exist_ok=True)
    arms = []
    for d in dirs:
        for arm_dir in sorted(p for p in d.iterdir() if p.is_dir() and (p / 'results.json').is_file()):
            if arm_dir.name not in arms:
                arms.append(arm_dir.name)
    summaries, task_sets = [], {}
    for arm in arms:
        sources = [d / arm / 'results.json' for d in dirs if (d / arm / 'results.json').is_file()]
        merged = merge_results(sources)
        arm_out = out / arm
        arm_out.mkdir(exist_ok=True)
        (arm_out / 'results.json').write_text(json.dumps(merged, default=str))
        results = Results.model_validate_json((arm_out / 'results.json').read_text())
        steps = {}
        for d in dirs:
            for row in read_jsonl(d / arm / 'steps.jsonl'):
                steps.setdefault(str(row.get('episode')), []).append(row)
        keep = {str(s.get('task_id')) for s in merged['simulations'] if _completed(s)}
        rows = [r for t, rs in steps.items() for r in rs if t in keep]
        with (arm_out / 'steps.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + '\n')
        for name in ('ledgers.json', 'agents.json'):
            merged_json = merge_keyed_json([d / arm / name for d in dirs])
            if merged_json:
                (arm_out / name).write_text(json.dumps(merged_json, default=str))
        summary = summarize_results(results)
        summary.update({'arm': arm, 'expected': len(merged['simulations']), 'routing': route_metrics(rows), 'merged_from': [str(d) for d in dirs]})
        if rows:
            t = max(len(rows), 1)
            summary.update({'jev_calls_per_turn': sum(r.get('jev_calls') or 0 for r in rows) / t, 'llm_calls_per_turn': sum(r.get('llm_calls') or 0 for r in rows) / t,
                            'turn_model_latency_ms_median': None, 'turn_sandbox_ms_median': None})
        elif arm != 'llm_only':
            summary.update({'jev_calls_per_turn': None, 'llm_calls_per_turn': None, 'turn_model_latency_ms_median': None, 'turn_sandbox_ms_median': None})
        summaries.append(summary)
        task_sets[arm] = keep
    common = set.intersection(*task_sets.values()) if task_sets else set()
    for s in summaries:
        arm = s['arm']
        merged = json.loads((out / arm / 'results.json').read_text())
        on_common = [float(sim['reward_info']['reward']) for sim in merged['simulations'] if _completed(sim) and str(sim['task_id']) in common]
        s['pass1_common'] = (sum(on_common) / len(on_common)) if on_common else None
        s['n_common'] = len(on_common)
    n = min((s['completed'] for s in summaries), default=0)
    h = half_width(len(common)) if common else None
    note = (f'Merged from {len(dirs)} runs. Common completed tasks across arms: {len(common)}; 95% half-width on pass^1 at that n (p=0.8) is '
            + (f'±{h:.1f} pt. ' if h else 'undefined. ') + 'pass1_common in summary.json is the comparison on the common set.')
    decisions = decide(summaries, h) if h else ['no common tasks']
    (out / 'summary.json').write_text(json.dumps({'summaries': summaries, 'decisions': decisions, 'resolution_note': note, 'common_tasks': sorted(common, key=lambda t: (len(t), t))}, indent=1, default=str))
    (out / 'manifest.json').write_text(json.dumps({'merged_from': [str(d) for d in dirs], 'arms': arms, 'created': dt.datetime.now().isoformat()}, indent=1))
    write_table(summaries, out / 'table.md', note)
    print((out / 'table.md').read_text())
    for line in decisions:
        print('DECISION:', line)
    print('common-set pass^1:', {s['arm']: (round(s['pass1_common'], 3) if s['pass1_common'] is not None else None, s['n_common']) for s in summaries})
    print(f'artifacts: {out}')


if __name__ == '__main__':
    main()
