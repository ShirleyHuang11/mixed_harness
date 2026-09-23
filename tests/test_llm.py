import json

import pytest

from auto_jev.llm import LLMClient, LLMProviderError, extract_json

SCHEMA = {'type': 'object', 'required': ['label'], 'properties': {'label': {'type': 'string', 'enum': ['a', 'b']}}}


def test_mock_builds_minimal_schema_instance_and_counts():
    client = LLMClient(kind='mock')
    result = client.complete('choose', output_schema=SCHEMA)
    assert result['value'] == {'label': 'a'}
    assert json.loads(result['text']) == {'label': 'a'}
    assert result['cache_hit'] is False and result['cost_usd'] is None
    assert client.stats['calls'] == 1 and client.stats['unpriced_calls'] == 0
    assert client.metadata['transport'] == 'mock' and client.mock is True


def test_mock_without_schema_returns_text():
    result = LLMClient(kind='mock').complete('first line\nlast line')
    assert result['value'] == result['text'] == 'last line'


def test_mock_responder_output_is_schema_checked():
    client = LLMClient(kind='mock', mock_responder=lambda prompt, schema: '```json\n{"label": "zzz"}\n```')
    with pytest.raises(LLMProviderError, match='enum'):
        client.complete('x', output_schema=SCHEMA)


def test_fenced_and_surrounded_json_is_extracted():
    assert extract_json('Sure:\n```json\n{"a": 1}\n```\nthanks') == '{"a": 1}'
    assert extract_json('prefix {"a": [1, 2]} suffix') == '{"a": [1, 2]}'


def test_cache_roundtrip_and_cache_only(tmp_path):
    responses = iter(['{"label": "b"}', '{"label": "a"}'])
    client = LLMClient(kind='mock', cache_dir=tmp_path, cache_namespace='ns', mock_responder=lambda p, s: next(responses))
    first = client.complete('same', output_schema=SCHEMA)
    second = client.complete('same', output_schema=SCHEMA)
    assert first['value'] == second['value'] == {'label': 'b'}
    assert second['cache_hit'] is True and client.stats['cache_hits'] == 1
    replay = LLMClient(kind='mock', cache_dir=tmp_path, cache_namespace='ns', cache_only=True)
    assert replay.complete('same', output_schema=SCHEMA)['value'] == {'label': 'b'}
    with pytest.raises(LLMProviderError, match='cache-only'):
        replay.complete('different', output_schema=SCHEMA)


def test_cache_integrity_failure_is_detected(tmp_path):
    client = LLMClient(kind='mock', cache_dir=tmp_path, cache_namespace='ns')
    client.complete('p')
    path = next((tmp_path / 'ns').glob('*.json'))
    tampered = json.loads(path.read_text()); tampered['text'] = 'edited'; path.write_text(json.dumps(tampered))
    with pytest.raises(LLMProviderError, match='integrity'):
        client.complete('p')


def test_real_kinds_require_key_and_never_call_without_it(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(LLMProviderError, match='ANTHROPIC_API_KEY'):
        LLMClient(kind='anthropic', model='claude-sonnet-5').complete('x')
    with pytest.raises(ValueError):
        LLMClient(kind='unknown')
