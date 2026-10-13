"""Calibrate a judge from a stage-0 replay: the reliability table, a fitted temperature, and the kill test.

    ~/tau2-bench/.venv/bin/python scripts/calibrate_judge.py --replay runs/stage0/<stamp>/replay.jsonl --out runs/stage0/<stamp>/judge.json

The replay holds, per closed turn, the judge's action probabilities (`judge_probs`, and for the logprob judge the
raw label log-probabilities `judge_logprobs`) and whether its pick was the LLM's action (`jev_correct`). A judge is
useful to the harness when its probability means what it says: in the bin of turns where it reports >= 0.99 the
pick should match the LLM on about 98% of them (Jev's number). This script

  1. prints the reliability table (probability bin -> agreement, n) and the expected calibration error (ECE);
  2. fits one temperature T on the raw log-probabilities (grid search, negative log-likelihood of the LLM's action
     among the options) and prints the table again after T;
  3. writes judge.json = {judge, temperature, reliability_before, reliability_after, ece_before, ece_after,
     verdict} and `replay.calibrated.jsonl`, the same rows with `jev_confidence` recomputed under T, which
     `scripts/fit_policy.py --replay` reads to fit the threshold table for this judge.

Kill test, written before the first open-model judge was run (2026-10-13): after calibration the >= 0.99 bin must
agree with the LLM on >= 85% of its turns and hold >= 10% of closed turns; below that the judge cannot be handed
turns by threshold and the verdict is "not usable as a judge".
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixedharness.store import read_jsonl  # noqa: E402

BINS = [('>=0.99', 0.99, 1.01), ('0.90-0.99', 0.90, 0.99), ('0.60-0.90', 0.60, 0.90), ('<0.60', -0.01, 0.60)]
KILL_TOP_BIN_AGREEMENT = 0.85
KILL_TOP_BIN_SHARE = 0.10
GRID = [round(0.2 + 0.05 * i, 2) for i in range(97)]   # 0.2 .. 5.0


def softmax(logprobs, t):
    top = max(logprobs.values())
    w = {k: math.exp((v - top) / t) for k, v in logprobs.items()}
    z = sum(w.values())
    return {k: x / z for k, x in w.items()}


def probs_under(row, t):
    """The judge's option probabilities under temperature t: from raw label log-probabilities when the replay has them."""
    lp = row.get('judge_logprobs')
    if lp and lp.get('labels') and lp.get('ids'):
        labels, ids = lp['labels'], lp['ids']
        letters = [chr(ord('A') + i) for i in range(len(ids))]
        p = softmax({l: labels.get(l, -30.0) for l in letters}, t)
        return {oid: p[l] for oid, l in zip(ids, letters)}
    return row.get('judge_probs')


def reliability(rows, confidence):
    """rows -> [{bin, n, share, agreement}] on closed turns with a target; `confidence(row)` gives the top probability."""
    scored = [r for r in rows if r.get('label_closed') and r.get('jev_correct') is not None and confidence(r) is not None]
    out, ece = [], 0.0
    for name, lo, hi in BINS:
        sub = [r for r in scored if lo <= confidence(r) < hi]
        if sub:
            agree = sum(1 for r in sub if r['jev_correct']) / len(sub)
            mean_conf = sum(confidence(r) for r in sub) / len(sub)
            ece += len(sub) / len(scored) * abs(mean_conf - agree)
        else:
            agree, mean_conf = None, None
        out.append({'bin': name, 'n': len(sub), 'share': len(sub) / len(scored) if scored else 0.0, 'agreement': agree, 'mean_confidence': mean_conf})
    return out, ece, len(scored)


def nll(rows, t):
    total, n = 0.0, 0
    for r in rows:
        if not r.get('label_closed') or not r.get('target') or not r.get('judge_logprobs'):
            continue
        p = probs_under(r, t)
        if p is None or r['target'] not in p:
            continue
        total += -math.log(max(p[r['target']], 1e-9))
        n += 1
    return total / n if n else float('inf'), n


def render(table, ece, n):
    lines = ['| judge probability | turns | share | agreement with the LLM |', '|---|---|---|---|']
    for b in table:
        lines.append(f"| {b['bin']} | {b['n']} | {b['share']:.0%} | {('%.0f%%' % (100 * b['agreement'])) if b['agreement'] is not None else '—'} |")
    lines.append(f'ECE {ece:.3f} on {n} closed turns')
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--replay', required=True)
    p.add_argument('--out', default=None, help='judge.json path; default next to the replay')
    args = p.parse_args(argv)
    rows = read_jsonl(args.replay)
    judge = next((r.get('judge') for r in rows if r.get('judge')), 'jev')
    before, ece_before, n = reliability(rows, lambda r: r.get('jev_confidence'))
    print(f'judge {judge}: reliability as reported')
    print(render(before, ece_before, n))
    temperature = 1.0
    fitted = [r for r in rows if r.get('judge_logprobs')]
    if fitted:
        scores = {t: nll(rows, t)[0] for t in GRID}
        temperature = min(scores, key=scores.get)
        print(f'\nfitted temperature T = {temperature} (NLL {scores[temperature]:.3f} vs {scores[1.0]:.3f} at T=1) on {nll(rows, temperature)[1]} turns')
    else:
        print('\nno raw log-probabilities in this replay (Jev replays carry none): temperature stays 1.0')

    def calibrated_conf(r):
        p = probs_under(r, temperature)
        return max(p.values()) if p else r.get('jev_confidence')

    after, ece_after, _ = reliability(rows, calibrated_conf)
    print(f'\nreliability after T = {temperature}')
    print(render(after, ece_after, n))
    top = after[0]
    verdict = 'usable' if (top['agreement'] is not None and top['agreement'] >= KILL_TOP_BIN_AGREEMENT and top['share'] >= KILL_TOP_BIN_SHARE) else 'not usable as a judge'
    print(f"\nverdict: {verdict} (>= 0.99 bin: agreement {('%.0f%%' % (100 * top['agreement'])) if top['agreement'] is not None else '—'}, share {top['share']:.0%}; "
          f'kill below {KILL_TOP_BIN_AGREEMENT:.0%} agreement or {KILL_TOP_BIN_SHARE:.0%} share)')
    out = Path(args.out) if args.out else Path(args.replay).with_name('judge.json')
    out.write_text(json.dumps({'judge': judge, 'temperature': temperature, 'replay': args.replay, 'closed_turns': n,
                               'reliability_before': before, 'ece_before': ece_before, 'reliability_after': after, 'ece_after': ece_after,
                               'verdict': verdict, 'kill': {'top_bin_agreement': KILL_TOP_BIN_AGREEMENT, 'top_bin_share': KILL_TOP_BIN_SHARE}}, indent=1))
    calibrated = Path(args.replay).with_name(Path(args.replay).stem + '.calibrated.jsonl')
    with calibrated.open('w') as f:
        for r in rows:
            probs = probs_under(r, temperature)
            if probs and r.get('jev_action') in probs:
                r = dict(r, jev_confidence=probs[r['jev_action']], judge_probs=probs, judge_temperature=temperature)
            f.write(json.dumps(r, default=str) + '\n')
    print(f'written: {out}\nwritten: {calibrated}')


if __name__ == '__main__':
    main()
