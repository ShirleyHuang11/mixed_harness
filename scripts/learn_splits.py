"""Offline: can a finer state type make the LLM-fallback steps deterministic?

    python scripts/learn_splits.py runs/tau2/<run>/routed/ledgers.json [...] [--min-n 10] [--purity 0.9] [--test <ledgers.json>]

For every step that went to the LLM (path_source low_confidence) the ledger holds the observation (conversation,
incoming event), the harness memory after the step and the action the LLM took. The compiler's own
`state_features` (examples/tau2/seeds.py, COMMON_SOURCE) is run on each, so a row learned here keys on exactly what
the compiler computes at runtime. Per state type (cluster) the learner searches single features and feature pairs
whose subset has >= min_n steps and one action class >= purity of the time; a greedy largest-first cover of those
subsets is the list of split rows for policy["splits"]. Printed: the rows, the share of all LLM steps they cover
(the predicted move of traffic from the LLM to code) and, with --test, the held-out purity on other ledgers.

Two features, `pending` and `n_completed`, are excluded when derived from a ledger that recorded memory after the
decision node (those change inside the step and would leak the answer); ledgers that carry routing.features use
all of them.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.tau2 import seeds  # noqa: E402

_NS = {}
exec(seeds.COMMON_SOURCE, _NS)   # the compiler's helpers, including state_features
LEAKY = ('pending', 'n_completed')


def features_of(entry):
    r = entry.get('routing') or {}
    if r.get('features'):
        return dict(r['features']), False
    o = entry['observation']; mem = (entry.get('decision') or {}).get('memory') or {}
    conv = list(o.get('conversation') or []); inc = o.get('incoming') or {}
    if inc.get('role') == 'user' and (not conv or conv[-1].get('text') != inc.get('text')):
        conv.append(inc)
    user_texts = [m.get('text') or '' for m in conv if m.get('role') == 'user']
    last_user = user_texts[-1] if user_texts else ''
    user = mem.get('user') or {}
    f = _NS['state_features'](inc, user_texts, last_user, mem, list(user.get('orders') or []), list((mem.get('orders') or {}).keys()))
    return f, True


def action_class(entry):
    a = entry.get('action') or {}
    if a.get('kind') == 'tool':
        return 'tool:' + str(a.get('name'))
    text = (a.get('text') or '').strip()
    if re.search(r"\b(confirm|proceed|shall i|would you like me to|go ahead)\b", text, re.I) and text.endswith('?'):
        return 'message:confirm'
    if text.endswith('?'):
        return 'message:ask'
    return 'message:report'


def load_steps(paths):
    steps = []
    for path in paths:
        for key, entries in json.load(open(path)).items():
            for e in entries:
                r = e.get('routing') or {}
                if r.get('path_source') == 'low_confidence' and e.get('action') and e.get('observation'):
                    f, leaky = features_of(e)
                    if leaky:
                        for k in LEAKY:
                            f.pop(k, None)
                    steps.append({'cluster': r['cluster'], 'f': f, 'y': action_class(e)})
    return steps


def matches(step, row):
    return step['cluster'] == row['cluster'] and all(step['f'].get(k) == v for k, v in (row.get('split') or {}).items())


def fit_splits(steps, *, min_n=10, purity=0.9, max_features=2):
    by = defaultdict(list)
    for s in steps:
        by[s['cluster']].append(s)
    names = sorted({k for s in steps for k in s['f']})
    candidates = []
    for cluster, rows in by.items():
        if len(rows) < min_n:
            continue
        for k in range(0, max_features + 1):
            for combo in combinations(names, k):
                groups = defaultdict(list)
                for s in rows:
                    groups[tuple(s['f'].get(c) for c in combo)].append(s)
                for vals, g in groups.items():
                    if len(g) < min_n:
                        continue
                    top, n_top = Counter(s['y'] for s in g).most_common(1)[0]
                    if n_top / len(g) >= purity and top != 'message:ask':
                        candidates.append({'cluster': cluster, 'split': dict(zip(combo, vals)), 'n': len(g), 'action': top, 'purity': round(n_top / len(g), 3)})
    chosen, covered = [], set()
    for c in sorted(candidates, key=lambda c: (-c['n'], len(c['split']))):
        ids = {id(s) for s in by[c['cluster']] if matches(s, c)}
        if len(ids - covered) < min_n:
            continue
        c['n_new'] = len(ids - covered)
        covered |= ids
        chosen.append(c)
    return {'rows': chosen, 'n_steps': len(steps), 'covered': len(covered)}


def evaluate(rows, steps):
    seen, hit, tot = set(), 0, 0
    for row in rows:
        rs = [s for s in steps if id(s) not in seen and matches(s, row)]
        seen |= {id(s) for s in rs}
        row['test_n'] = len(rs)
        row['test_purity'] = round(sum(1 for s in rs if s['y'] == row['action']) / len(rs), 3) if rs else None
        hit += sum(1 for s in rs if s['y'] == row['action']); tot += len(rs)
    return {'n_steps': len(steps), 'covered': tot, 'agreement': (hit / tot) if tot else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ledgers', nargs='+')
    p.add_argument('--min-n', type=int, default=10)
    p.add_argument('--purity', type=float, default=0.9)
    p.add_argument('--test', nargs='*', default=[], help='held-out ledgers.json files')
    p.add_argument('--out', default=None)
    args = p.parse_args(argv)
    steps = load_steps(args.ledgers)
    fit = fit_splits(steps, min_n=args.min_n, purity=args.purity)
    print(f"LLM-fallback steps: {fit['n_steps']}; split rows: {len(fit['rows'])}; predicted move to code: {fit['covered']} ({fit['covered'] / max(1, fit['n_steps']):.1%})")
    if args.test:
        ev = evaluate(fit['rows'], load_steps(args.test))
        print(f"held-out: covered {ev['covered']} of {ev['n_steps']} ({ev['covered'] / max(1, ev['n_steps']):.1%}); row action = LLM action {ev['agreement']:.1%}" if ev['agreement'] is not None else 'held-out: nothing covered')
    print(f"{'cluster':40s} {'split':50s} {'n':>4s} {'pur':>5s} {'test':>9s} action")
    for c in fit['rows']:
        t = f"{c.get('test_n', '')}/{c.get('test_purity', '')}" if 'test_n' in c else ''
        print(f"{c['cluster'][:40]:40s} {str(c['split'])[:50]:50s} {c['n']:4d} {c['purity']:5.2f} {t:>9s} {c['action']}")
    print('actions of covered steps:', Counter(c['action'] for c in fit['rows'] for _ in range(c['n_new'])).most_common(6))
    if args.out:
        json.dump(fit, open(args.out, 'w'), indent=1)
        print('written:', args.out)


if __name__ == '__main__':
    main()
