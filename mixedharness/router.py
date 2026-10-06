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
PATH_SOURCES = ('jev', 'code_override', 'low_confidence', 'escalate_below_tau', 'tool_result', 'fallthrough', 'policy', 'shadow', 'trajectory', 'forced', 'guard')
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


TAU_GRID = (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
NEVER = 1.01   # a tau no confidence reaches: the cluster always escalates


def fit_tau(replay_rows, *, target=0.9, min_n=10, grid=TAU_GRID, default=0.9):
    """Loop B, the threshold per cluster, from stage-0 replay rows (label_closed, cluster, jev_confidence, jev_correct).

    For each cluster with at least `min_n` closed-labelled turns, the smallest tau in `grid` at which Jev's
    agreement with the LLM on the turns at or above tau is at least `target` (and at least min_n // 2 turns are
    accepted). A cluster where no tau reaches the target gets NEVER (always escalate). Clusters with too few
    turns are absent and fall back to the global default at runtime. Returns {'target', 'min_n', 'default',
    'clusters': {cluster: {'tau', 'n', 'accepted', 'agreement'}}}."""
    by = {}
    for r in replay_rows:
        if r.get('label_closed') and r.get('cluster') and r.get('jev_correct') is not None:
            by.setdefault(r['cluster'], []).append(r)
    out = {}
    for cluster, rows in sorted(by.items()):
        if len(rows) < min_n:
            continue
        chosen = None
        for tau in grid:
            acc = [r for r in rows if (r.get('jev_confidence') or 0.0) >= tau]
            if len(acc) >= max(1, min_n // 2):
                agreement = sum(1 for r in acc if r['jev_correct']) / len(acc)
                if agreement >= target:
                    chosen = {'tau': tau, 'n': len(rows), 'accepted': len(acc), 'agreement': agreement}
                    break
        out[cluster] = chosen or {'tau': NEVER, 'n': len(rows), 'accepted': 0, 'agreement': None}
    return {'target': target, 'min_n': min_n, 'default': default, 'clusters': out}


def fit_from_steps(steps, *, min_n=10, same_action=0.95, lost_ratio=1.5):
    """Loop B on routed step tables: which state types can move to code, which must tighten.

    A cluster moves to code (a `default` action) when Jev's accepted picks there were the same option at least
    `same_action` of the time over at least `min_n` steps and those steps' episodes were won at least as often as
    the overall rate. A cluster tightens to NEVER when its accepted Jev steps sit in lost episodes at more than
    `lost_ratio` times the overall lost share (over at least `min_n` steps). Returns {'defaults': {cluster: option},
    'tighten': [cluster], 'stats': {cluster: {...}}}."""
    scored = [s for s in steps if s.get('reward') is not None and s.get('cluster')]
    if not scored:
        return {'defaults': {}, 'tighten': [], 'stats': {}}
    lost_overall = sum(1 for s in scored if s['reward'] == 0.0) / len(scored)
    won_overall = 1.0 - lost_overall
    by = {}
    for s in scored:
        if s.get('path') == 'closed' and s.get('path_source') in ('jev', 'structural', 'policy'):
            by.setdefault(s['cluster'], []).append(s)
    defaults, tighten, stats = {}, [], {}
    for cluster, rows in sorted(by.items()):
        if len(rows) < min_n:
            continue
        picks = {}
        for r in rows:
            picks[r.get('chosen')] = picks.get(r.get('chosen'), 0) + 1
        top, n_top = max(picks.items(), key=lambda kv: kv[1])
        lost = sum(1 for r in rows if r['reward'] == 0.0) / len(rows)
        stats[cluster] = {'n': len(rows), 'top': top, 'top_share': n_top / len(rows), 'lost_share': lost}
        if top and n_top / len(rows) >= same_action and (1 - lost) >= won_overall - 0.05 and not str(top).startswith('menu:') and not str(top).startswith('agent:'):
            defaults[cluster] = top
        jev_rows = [r for r in rows if r.get('path_source') == 'jev']
        if len(jev_rows) >= min_n and lost_overall > 0 and (sum(1 for r in jev_rows if r['reward'] == 0.0) / len(jev_rows)) > lost_ratio * lost_overall:
            tighten.append(cluster)
    return {'defaults': defaults, 'tighten': tighten, 'stats': stats}


def compact_policy(policy):
    """The part of a fitted policy a harness needs at runtime: preferred path per cluster, tau per cluster, epsilon."""
    tau = {c: float(v['tau']) for c, v in (policy.get('tau_by_cluster') or {}).items() if isinstance(v, dict) and v.get('tau') is not None}
    for c in policy.get('tighten') or []:
        tau[c] = NEVER
    return {'epsilon': float(policy.get('epsilon', DEFAULT_EPSILON)),
            'prefer': {c: v['prefer'] for c, v in (policy.get('clusters') or {}).items() if isinstance(v, dict) and v.get('prefer')},
            'tau': tau, 'defaults': dict(policy.get('defaults') or {})}


def merge_policy(path_policy=None, tau_policy=None, *, epsilon=DEFAULT_EPSILON, learned=None):
    """One policy from a fit_policy result (paths), a fit_tau result (thresholds) and a fit_from_steps result (code rows)."""
    return {'epsilon': epsilon, 'clusters': (path_policy or {}).get('clusters') or {},
            'tau_by_cluster': (tau_policy or {}).get('clusters') or {},
            'defaults': (learned or {}).get('defaults') or {}, 'tighten': (learned or {}).get('tighten') or []}


def policy_source_literal(policy):
    """JSON literal for embedding in a python node's source (the seed replaces a placeholder with it).
    Accepts a full fitted policy or an already compact one ({'epsilon', 'prefer', 'tau'})."""
    if not policy:
        return json.dumps({'epsilon': 0.0, 'prefer': {}, 'tau': {}, 'defaults': {}}, sort_keys=True)
    if 'clusters' in policy or 'tau_by_cluster' in policy or 'tighten' in policy:
        policy = compact_policy(policy)
    return json.dumps({'epsilon': float(policy.get('epsilon', 0.0)), 'prefer': dict(policy.get('prefer') or {}), 'tau': dict(policy.get('tau') or {}),
                       'defaults': dict(policy.get('defaults') or {})}, sort_keys=True)


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
