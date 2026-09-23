import pytest

from auto_jev.budget import check_budget, episode_usage, validate_budget


def result_with(latencies, node_costs=()):
    traces = [{'elapsed_ms': ms, 'trace': [{'kind': 'llm', 'response': {'cost_usd': cost}} for cost in node_costs]} for ms in latencies]
    return {'traces': traces}


def test_usage_percentiles_and_costs():
    usage = episode_usage(result_with([100, 200, 300, 400, 1000], node_costs=[0.01, None]))
    assert usage['decisions'] == 5 and usage['latency_ms_p95'] == 1000 and usage['latency_ms_median'] == 300
    assert usage['cost_usd'] == pytest.approx(0.05) and usage['unpriced_calls'] == 5


def test_usage_without_traces():
    assert episode_usage({}) == {'latency_ms_p95': None, 'latency_ms_median': None, 'decisions': 0, 'cost_usd': None, 'unpriced_calls': 0}


def test_jev_gateway_cost_is_read():
    trace = [{'kind': 'jev', 'response': {'gateway_metadata': {'provider_metadata': {'gateway': {'cost': '0.002'}}}}}]
    assert episode_usage({'traces': [{'elapsed_ms': 5, 'trace': trace}]})['cost_usd'] == pytest.approx(0.002)


def test_check_budget_reports_violations_only_when_measurable():
    usage = episode_usage(result_with([100, 900], node_costs=[0.5]))
    assert check_budget(usage, {'latency_ms_p95': 500, 'cost_usd_per_episode': 0.1}) == [
        'latency_ms_p95 900.0 > 500', 'cost_usd_per_episode 1.0 > 0.1']
    assert check_budget(usage, {'latency_ms_p95': 1000, 'cost_usd_per_episode': 2}) == []
    assert check_budget(episode_usage({}), {'latency_ms_p95': 1}) == []


@pytest.mark.parametrize('bad', [{'latency_ms_p95': -1}, {'unknown': 1}, {'latency_ms_p95': 'fast'}, 'budget'])
def test_validate_budget_rejects(bad):
    with pytest.raises(ValueError):
        validate_budget(bad)
