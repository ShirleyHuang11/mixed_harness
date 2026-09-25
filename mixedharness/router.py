"""The routing layer: the three paths, the routing signals, and the per-cluster policy learned from steps.

A step is routed to one of three paths:

    closed    Jev picks from the compiled action set; code fills the arguments; a template or slot filler writes text
    narrow    the LLM writes a short menu of candidate actions; Jev picks one
    escalate  the LLM takes the whole step (one native tool-calling turn)

Three signals feed the decision, in this order of authority:

  1. the compiler's `closed` flag (code): the legal action set is non-empty and every argument of every option can
     be filled from state. When it is false the step cannot be closed, whatever Jev says.
  2. Jev's own route pick, asked in the same request as the action pick; a closed pick is accepted only when the
     action pick's confidence clears tau (0.9 from stage 0, where every pick at or above it agreed with the LLM and
     6 of 15 below it did): below tau the step escalates; a closed pick whose reordered paraphrase disagrees goes narrow.
  3. a per-cluster policy learned from recorded steps (`fit_policy`): the path with the best mean of
     success - lambda * cost in that cluster, with an epsilon share of shadow runs on the other paths.

Only (3) lives in this module as learning code; (1) and (2) are computed inside the harness's own nodes, because
python nodes run sandboxed and cannot import anything. `policy_source_literal` renders a fitted policy as the JSON
literal a seed embeds, so the policy is part of the harness version and hashes with it.
"""
import hashlib
import json
import statistics

PATHS = ('closed', 'narrow', 'escalate')
PATH_SOURCES = ('jev', 'code_override', 'low_confidence', 'fallthrough', 'policy', 'shadow', 'trajectory', 'forced', 'guard')
DEFAULT_LAMBDA = 0.5          # success units per dollar of agent cost per step, scaled by COST_SCALE
COST_SCALE = 100.0            # reward = success - lambda * cost_usd * COST_SCALE  (one cent of LLM cost = 0.005 success)
MIN_SAMPLES = 8               # a path needs this many scored steps in a cluster before the policy will prefer it
DEFAULT_EPSILON = 0.1


def shadow_unit(*parts):
    """Deterministic pseudo-random unit in [0, 1) from the step identity; the same step always draws the same number."""
    digest = hashlib.sha256('|'.join(str(p) for p in parts).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / float(1 << 64)


def agreement(answers, primary='action', alternate='action_alt'):
    """Do the action question and its paraphrase pick the same option? Returns (agree, picks)."""
    a = ((answers.get(primary) or {}).get('choice'))
    b = ((answers.get(alternate) or {}).get('choice'))
    if a is None or b is None:
        return None, {primary: a, alternate: b}
    return a == b, {primary: a, alternate: b}


def step_reward(step, lam=DEFAULT_LAMBDA):
    """success - lambda * scaled cost, on a step whose episode reward is known."""
    reward = step.get('reward')
    if reward is None:
        return None
    return float(reward) - lam * float(step.get('cost_usd') or 0.0) * COST_SCALE


def fit_policy(steps, *, lam=DEFAULT_LAMBDA, min_samples=MIN_SAMPLES, epsilon=DEFAULT_EPSILON):
    """Loop B: per-cluster path preference from recorded steps with episode rewards attached.

    steps: rows from the step store (see trace.py) with `cluster`, `path` (the realised path) and `reward`.
    Returns {'lambda', 'epsilon', 'min_samples', 'clusters': {cluster: {'prefer': path|None, 'stats': {path: {n, mean}}}}}.
    A cluster prefers a path only when that path has at least `min_samples` scored steps and the best mean; with
    fewer samples the policy stays silent and the harness routes on its own signals.
    """
    by_cluster = {}
    for step in steps:
        r = step_reward(step, lam)
        if r is None or step.get('cluster') is None or step.get('path') not in PATHS:
            continue
        by_cluster.setdefault(step['cluster'], {}).setdefault(step['path'], []).append(r)
    clusters = {}
    for cluster, paths in sorted(by_cluster.items()):
        stats = {path: {'n': len(v), 'mean': statistics.mean(v)} for path, v in paths.items()}
        eligible = {p: s for p, s in stats.items() if s['n'] >= min_samples}
        prefer = max(eligible, key=lambda p: eligible[p]['mean']) if eligible else None
        clusters[cluster] = {'prefer': prefer, 'stats': stats}
    return {'lambda': lam, 'epsilon': epsilon, 'min_samples': min_samples, 'clusters': clusters}


def compact_policy(policy):
    """The part of a fitted policy a harness needs at runtime: {cluster: preferred path} plus epsilon."""
    return {'epsilon': float(policy.get('epsilon', DEFAULT_EPSILON)),
            'prefer': {c: v['prefer'] for c, v in (policy.get('clusters') or {}).items() if v.get('prefer')}}


def policy_source_literal(policy):
    """JSON literal for embedding in a python node's source (the seed replaces a placeholder with it)."""
    return json.dumps(compact_policy(policy) if policy else {'epsilon': 0.0, 'prefer': {}}, sort_keys=True)


def route_metrics(steps):
    """Loop-B facing metrics over step rows: path mix, traffic share, fall-through and override rates."""
    n = len(steps)
    if not n:
        return {'steps': 0}
    mix = {p: sum(1 for s in steps if s.get('path') == p) / n for p in PATHS}
    sources = {}
    for s in steps:
        sources[s.get('path_source')] = sources.get(s.get('path_source'), 0) + 1
    agree = [s['agreement'] for s in steps if s.get('agreement') is not None]
    closed_flag = [bool(s.get('closed')) for s in steps if s.get('closed') is not None]
    return {'steps': n, 'path_mix': mix, 'jev_traffic_share': mix['closed'],
            'llm_share': mix['narrow'] + mix['escalate'],
            'path_sources': {k: v / n for k, v in sources.items()},
            'agreement_rate': (sum(agree) / len(agree)) if agree else None,
            'compiler_closed_rate': (sum(closed_flag) / len(closed_flag)) if closed_flag else None,
            'invalid_actions': sum(1 for s in steps if s.get('invalid')),
            'guard_interventions': sum(1 for s in steps if s.get('guard') not in (None, 'ok'))}
