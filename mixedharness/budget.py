"""Per-episode usage read from complete traces, and explicit budget checks.

Usage is never estimated: latency comes from each decision's `elapsed_ms`, cost from what the providers
reported. A call whose provider reported no price is counted in `unpriced_calls`, not priced by guesswork.
"""
import math

BUDGET_KEYS = ('latency_ms_p95', 'cost_usd_per_episode')


class BudgetError(ValueError):
    pass


def validate_budget(budget):
    if not isinstance(budget, dict):
        raise BudgetError('budget must be an object')
    if set(budget) - set(BUDGET_KEYS):
        raise BudgetError(f'unknown budget keys {sorted(set(budget) - set(BUDGET_KEYS))}')
    for key, value in budget.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise BudgetError(f'budget {key} must be a non-negative finite number')
    return dict(budget)


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def episode_usage(result):
    """{latency_ms_p95, latency_ms_median, decisions, cost_usd|None, unpriced_calls} from result['traces']."""
    traces = result.get('traces') if isinstance(result, dict) else None
    traces = [t for t in (traces or []) if isinstance(t, dict)]
    latencies = [float(t['elapsed_ms']) for t in traces
                 if isinstance(t.get('elapsed_ms'), (int, float)) and not isinstance(t.get('elapsed_ms'), bool)]
    cost, unpriced, priced = 0.0, 0, False
    for decision in traces:
        for node in decision.get('trace') or []:
            if not isinstance(node, dict) or node.get('kind') not in ('jev', 'llm') or node.get('status') != 'ok':
                continue
            response = node.get('response') or {}
            node_cost = response.get('cost_usd') if isinstance(response, dict) else None
            if node_cost is None:
                unpriced += 1
            else:
                cost += float(node_cost)
                priced = True
    return {'latency_ms_p95': percentile(latencies, 0.95) if latencies else None,
            'latency_ms_median': percentile(latencies, 0.5) if latencies else None,
            'decisions': len(traces), 'cost_usd': cost if priced else None, 'unpriced_calls': unpriced}


def check_budget(usage, budget):
    """List of violation strings; empty when within budget or when the quantity is unknown."""
    budget = validate_budget(budget)
    violations = []
    if 'latency_ms_p95' in budget and usage.get('latency_ms_p95') is not None and usage['latency_ms_p95'] > budget['latency_ms_p95']:
        violations.append(f"latency_ms_p95 {usage['latency_ms_p95']:.0f} > {budget['latency_ms_p95']}")
    if 'cost_usd_per_episode' in budget and usage.get('cost_usd') is not None and usage['cost_usd'] > budget['cost_usd_per_episode']:
        violations.append(f"cost_usd_per_episode {usage['cost_usd']:.4f} > {budget['cost_usd_per_episode']}")
    return violations
