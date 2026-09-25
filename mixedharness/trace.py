"""The trace store's step table: one row per agent step, flat, with the episode reward joined in afterwards.

    runs/<run>/<arm>/steps.jsonl

A row is what the evolution loops and the benchmark table read; nothing is summarised from memory. Rows are
built from an agent ledger (examples/tau2/agent.py writes one entry per turn with the complete runtime result)
by `steps_from_ledger`, and the episode reward is attached once the simulation is scored.

Row fields:
    episode, trial, turn, arm, spec_hash, harness_version
    cluster, closed, needs           the compiler's view of the state
    path_jev, path, path_source      Jev's route pick, the realised path, and why they differ
    agreement, confidence, fits      the routing signals
    action_kind, action_name, chosen the action taken (tool name or template id)
    guard, invalid                   what the guard did; the agent wrapper's validity verdict
    jev_calls, llm_calls, cost_usd, latency_ms, model_ms
    reward                           the episode's reward, or null until scored
"""
from .store import append_jsonl, read_jsonl


def _routing(decision):
    """The guard node's routing record, if the harness has one; else the closed-path defaults."""
    nodes = (decision or {}).get('nodes') or {}
    guard = nodes.get('guard') if isinstance(nodes.get('guard'), dict) else None
    if guard and isinstance(guard.get('routing'), dict):
        return guard['routing']
    route = nodes.get('route') if isinstance(nodes.get('route'), dict) else None
    if route:
        return {k: route.get(k) for k in ('cluster', 'closed', 'needs', 'path_jev', 'path', 'path_source', 'agreement', 'confidence', 'fits')}
    return {}


def steps_from_ledger(ledger, *, episode, arm, spec_hash=None, trial=0, harness_version=None, reward=None):
    rows = []
    for entry in ledger:
        decision = entry.get('decision') if isinstance(entry.get('decision'), dict) else {}
        routing = entry.get('routing') if isinstance(entry.get('routing'), dict) else _routing(decision)
        action = entry.get('action') or {}
        nodes = decision.get('nodes') or {}
        guard = nodes.get('guard') if isinstance(nodes.get('guard'), dict) else {}
        rows.append({'episode': episode, 'trial': trial, 'turn': entry.get('turn'), 'arm': arm, 'spec_hash': spec_hash,
                     'harness_version': harness_version,
                     'cluster': routing.get('cluster'), 'closed': routing.get('closed'), 'needs': routing.get('needs'),
                     'path_jev': routing.get('path_jev'), 'path': routing.get('path') or ('escalate' if arm == 'llm_only' else None),
                     'path_source': routing.get('path_source'),
                     'agreement': routing.get('agreement'), 'confidence': routing.get('confidence'), 'fits': routing.get('fits'),
                     'action_kind': action.get('kind'), 'action_name': action.get('name'), 'chosen': guard.get('chosen', entry.get('chosen')),
                     'guard': entry.get('guard', guard.get('guard')), 'invalid': entry.get('invalid'),
                     'jev_calls': entry.get('jev_calls'), 'llm_calls': entry.get('llm_calls'), 'cost_usd': entry.get('cost_usd'),
                     'latency_ms': entry.get('elapsed_ms'), 'model_ms': entry.get('model_ms'), 'reward': reward})
    return rows


def write_steps(path, rows):
    for row in rows:
        append_jsonl(path, row)


def read_steps(path):
    return read_jsonl(path)
