"""Per-episode usage derived from complete traces, and explicit budget checks."""
import math

BUDGET_KEYS = ('latency_ms_p95', 'cost_usd_per_episode')


def validate_budget(budget):
    if not isinstance(budget, dict):
        raise ValueError('budget must be an object')
    if set(budget) - set(BUDGET_KEYS):
        raise ValueError(f'unknown budget keys {sorted(set(budget) - set(BUDGET_KEYS))}')
    for key, value in budget.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'budget {key} must be a non-negative finite number')
    return dict(budget)


def _percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _node_cost(response):
    if not isinstance(response, dict):
        return None
    if response.get('cost_usd') is not None:
        return float(response['cost_usd'])
    gateway = (response.get('gateway_metadata') or {}).get('provider_metadata') or {}
    cost = (gateway.get('gateway') or {}).get('cost')
    return float(cost) if cost is not None else None


def episode_usage(result):
    traces = result.get('traces') if isinstance(result, dict) else None
    traces = traces if isinstance(traces, list) else []
    latencies = [float(t['elapsed_ms']) for t in traces
                 if isinstance(t, dict) and isinstance(t.get('elapsed_ms'), (int, float)) and not isinstance(t['elapsed_ms'], bool)]
    cost, unpriced, priced = 0.0, 0, False
    for decision in traces:
        for node in (decision.get('trace') or []) if isinstance(decision, dict) else []:
            if not isinstance(node, dict) or node.get('kind') not in ('jev', 'llm'):
                continue
            node_cost = _node_cost(node.get('response'))
            if node_cost is None:
                unpriced += 1
            else:
                cost += node_cost; priced = True
    return {'latency_ms_p95': _percentile(latencies, 0.95) if latencies else None,
            'latency_ms_median': _percentile(latencies, 0.5) if latencies else None,
            'decisions': len(traces), 'cost_usd': cost if priced else None, 'unpriced_calls': unpriced}


def check_budget(usage, budget):
    budget = validate_budget(budget)
    violations = []
    if 'latency_ms_p95' in budget and usage.get('latency_ms_p95') is not None and usage['latency_ms_p95'] > budget['latency_ms_p95']:
        violations.append(f"latency_ms_p95 {usage['latency_ms_p95']} > {budget['latency_ms_p95']}")
    if 'cost_usd_per_episode' in budget and usage.get('cost_usd') is not None and usage['cost_usd'] > budget['cost_usd_per_episode']:
        violations.append(f"cost_usd_per_episode {usage['cost_usd']} > {budget['cost_usd_per_episode']}")
    return violations
