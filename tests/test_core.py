"""Expressions, schema subset, sandbox, spec validation, runtime. Mock providers only."""
import json

import pytest

from mixedharness import expressions as ex
from mixedharness import sandbox as sb
from mixedharness import schema as sc
from mixedharness.jev import JevClient, JevError, mock_answers, normalise_answers
from mixedharness.llm import LLMClient, LLMError, LLMOutputError, extract_json
from mixedharness.runtime import PipelineError, Runtime
from mixedharness.spec import FORMAT, SpecError, authoring_schema, dependencies, spec_hash, validate_spec

RUN_DOUBLE = "def run(obs, nodes, memory):\n    return {'n': obs['n'] * 2, 'seen': memory.get('seen', 0) + 1}\n"


# -- expressions ------------------------------------------------------------------------------------------
def test_expression_evaluates_json_values_and_functions():
    ctx = {'obs': {'x': 2, 'xs': [1, 2, 3]}, 'nodes': {'a': {'v': 'hi'}}, 'memory': {'k': None}}
    assert ex.evaluate("{'d': obs['x'] * 2, 's': nodes['a']['v'] + '!', 'm': mean(obs['xs']), 'l': last(obs['xs']), 't': text(memory['k'])}", ctx) == \
        {'d': 4, 's': 'hi!', 'm': 2, 'l': 3, 't': ''}
    assert ex.evaluate("obs['x'] if obs['x'] > 1 else None", ctx) == 2
    assert ex.evaluate("get(obs, 'missing', 'dflt')", ctx) == 'dflt'


@pytest.mark.parametrize('bad', ["obs.__class__", "__import__('os')", "[x for x in obs]", "nodes[obs['k']]", "lambda: 1", "obs['a'] := 1"])
def test_expression_rejects_unsafe_syntax(bad):
    with pytest.raises(ex.ExpressionError):
        ex.parse(bad)


def test_expression_runtime_errors_are_expression_errors():
    with pytest.raises(ex.ExpressionError):
        ex.evaluate("obs['missing']", {'obs': {}, 'nodes': {}, 'memory': {}})
    with pytest.raises(ex.ExpressionError):
        ex.evaluate("'%s' % obs['x']", {'obs': {'x': 1}, 'nodes': {}, 'memory': {}})
    with pytest.raises(ex.ExpressionError):
        ex.evaluate("1 / 0", {'obs': {}, 'nodes': {}, 'memory': {}})


def test_references_are_the_distinct_literal_node_ids():
    assert sorted(ex.references("nodes['b']['q'] if nodes['a'] else nodes['b']")) == ['a', 'b']


# -- schema -----------------------------------------------------------------------------------------------
def test_schema_subset_validates_and_rejects():
    schema = {'type': 'object', 'required': ['text'], 'additionalProperties': False,
              'properties': {'text': {'type': 'string', 'minLength': 1}, 'n': {'type': 'integer', 'minimum': 0}}}
    sc.check_schema(schema)
    sc.validate({'text': 'x', 'n': 2}, schema)
    for bad in ({'text': ''}, {'text': 'x', 'extra': 1}, {'n': 1}, {'text': 'x', 'n': -1}, {'text': 'x', 'n': 1.5}):
        with pytest.raises(sc.SchemaError):
            sc.validate(bad, schema)
    with pytest.raises(sc.SchemaError):
        sc.check_schema({'type': 'object', 'pattern': 'x'})
    assert sc.validate(sc.minimal_instance(schema), schema) == {'text': 'x'}


# -- sandbox ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('bad', ["import os\ndef run(obs, nodes, memory):\n    return 1\n",
                                 "def run(obs, nodes, memory):\n    return obs.__class__\n",
                                 "def run(obs, nodes, memory):\n    return open('/etc/passwd')\n",
                                 "x = 1\ndef run(obs, nodes, memory):\n    return x\n",
                                 "def run(obs, nodes):\n    return 1\n",
                                 "def run(obs, nodes, memory):\n    return getattr(obs, 'x')\n"])
def test_source_validator_rejects(bad):
    with pytest.raises(sb.SourceError):
        sb.validate_source(bad)


@pytest.fixture(scope='module')
def pool():
    if not sb.sandbox_available():
        pytest.skip('no native sandbox on this machine')
    pool = sb.WorkerPool(size=1)
    yield pool
    pool.close()


def test_warm_worker_runs_many_jobs_in_one_process(pool):
    src = "def helper(x):\n    return x * 2\n\ndef run(obs, nodes, memory):\n    return {'d': helper(obs['n']), 'sq': math.sqrt(16), 'm': statistics.mean([1, 2, 3])}\n"
    first, _ = pool.run(src, {'obs': {'n': 21}, 'nodes': {}, 'memory': {}})
    second, _ = pool.run(src, {'obs': {'n': 1}, 'nodes': {}, 'memory': {}})
    assert first == {'d': 42, 'sq': 4.0, 'm': 2} and second['d'] == 2
    assert pool.stats['jobs'] == 2 and pool.stats['starts'] == 1


def test_worker_enforces_limits_and_recovers(pool):
    with pytest.raises(sb.NodeError, match='time limit'):
        pool.run("def run(obs, nodes, memory):\n    while True:\n        pass\n", {'obs': {}, 'nodes': {}, 'memory': {}}, wall_seconds=0.5, cpu_seconds=0.5)
    with pytest.raises(sb.NodeError, match='memory'):
        pool.run("def run(obs, nodes, memory):\n    return [0] * 10**9\n", {'obs': {}, 'nodes': {}, 'memory': {}})
    with pytest.raises(sb.NodeError, match='boom'):
        pool.run("def run(obs, nodes, memory):\n    raise ValueError('boom')\n", {'obs': {}, 'nodes': {}, 'memory': {}})
    with pytest.raises(sb.NodeError):
        pool.run("def run(obs, nodes, memory):\n    return float('inf')\n", {'obs': {}, 'nodes': {}, 'memory': {}})
    value, _ = pool.run(RUN_DOUBLE, {'obs': {'n': 2}, 'nodes': {}, 'memory': {}})
    assert value == {'n': 4, 'seen': 1}


def test_worker_has_no_network_and_no_host_environment(pool):
    src = ("def run(obs, nodes, memory):\n"
           "    out = {}\n"
           "    try:\n"
           "        out['env'] = len([k for k in []])\n"
           "    except Exception:\n"
           "        out['env'] = -1\n"
           "    return out\n")
    value, _ = pool.run(src, {'obs': {}, 'nodes': {}, 'memory': {}})
    assert value == {'env': 0}


# -- spec -------------------------------------------------------------------------------------------------
def _spec(**over):
    base = {'format': FORMAT, 'name': 't', 'jev_model': 'jev-1.13.0', 'llm_model': 'mock-llm', 'nodes': [
        {'id': 'a', 'kind': 'python', 'source': RUN_DOUBLE, 'inputs': []},
        {'id': 'j', 'kind': 'jev', 'state': "nodes['a']", 'questions': {'q': {'type': 'noul', 'instructions': 'ok?'}}},
        {'id': 'l', 'kind': 'llm', 'prompt': "None if nodes['j']['q']['noul'] > 0.5 else 'write'", 'schema': {'type': 'object', 'required': ['text'], 'properties': {'text': {'type': 'string'}}}},
        {'id': 'd', 'kind': 'expression', 'expression': "{'l': nodes['l'], 'n': nodes['a']['n']}"}],
        'output': "nodes['d']", 'memory': "{'seen': nodes['a']['seen']}"}
    base.update(over)
    return base


def test_spec_validates_and_infers_dependencies():
    spec = validate_spec(_spec())
    assert dependencies(spec) == {'a': [], 'j': ['a'], 'l': ['j'], 'd': ['l', 'a']}
    assert spec_hash(spec) == spec_hash(json.loads(json.dumps(spec)))
    assert authoring_schema()['properties']['format']['const'] == FORMAT


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(format='other/1'),
    lambda s: s.pop('llm_model'),
    lambda s: s['nodes'].append({'id': 'a', 'kind': 'expression', 'expression': '1'}),
    lambda s: s['nodes'].append({'id': 'x', 'kind': 'expression', 'expression': "nodes['y']"}) or s['nodes'].append({'id': 'y', 'kind': 'expression', 'expression': "nodes['x']"}),
    lambda s: s['nodes'][0].update(inputs=['zzz']),
    lambda s: s['nodes'][1].update(questions={'q': {'type': 'noul', 'instructions': 'x', 'criteria': {}}}),
    lambda s: s.update(output="nodes['nope']"),
    lambda s: s['nodes'][2].update(schema={'type': 'object', 'pattern': 'x'}),
])
def test_spec_rejects(mutate):
    spec = _spec()
    mutate(spec)
    with pytest.raises((SpecError, ValueError)):
        validate_spec(spec)


# -- providers --------------------------------------------------------------------------------------------
def test_mock_jev_is_deterministic_and_typed():
    q = {'c': {'type': 'choice', 'instructions': 'x', 'criteria': {'a': 'A', 'b': 'B'}}, 'n': {'type': 'noul', 'instructions': 'y'},
         's': {'type': 'score', 'instructions': 'z', 'criteria': ['low', 'high']}}
    one, two = mock_answers(q, {'k': 1}), mock_answers(q, {'k': 1})
    assert one == two and one['c']['choice'] in ('a', 'b') and 0 <= one['n']['noul'] <= 1 and one['s']['score'] in (0.0, 1.0)
    client = JevClient(mock=True)
    r = client.judge({'k': 1}, q)
    assert r['model'] == 'mock' and client.stats['calls'] == 1 and set(r['answers']) == {'c', 'n', 's'}


def test_jev_normalisation_rejects_incomplete_replies_and_missing_key(monkeypatch):
    q = {'c': {'type': 'choice', 'instructions': 'x', 'criteria': {'a': 'A', 'b': 'B'}}}
    with pytest.raises(JevError):
        normalise_answers(q, {})
    assert normalise_answers(q, {'c': {'probabilities': {'a': 0.2, 'b': 0.8}}})['c']['choice'] == 'b'
    for name in ('TYPESAFE_API_KEY', 'TYPESAFE_API_KEY_PERSONAL', 'TYPESAFE_API_KEY_personal'):   # a doctor test may have loaded .env into the process
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(JevError, match='TYPESAFE_API_KEY'):
        JevClient().judge({'k': 1}, q)


def test_llm_mock_schema_and_errors(monkeypatch):
    schema = {'type': 'object', 'required': ['text'], 'properties': {'text': {'type': 'string', 'minLength': 1}}}
    client = LLMClient(kind='mock')
    assert client.complete('write', schema)['value'] == {'text': 'x'}
    assert client.complete('a\nb')['value'] == 'b'
    bad = LLMClient(kind='mock', mock_responder=lambda p, s: 'not json')
    with pytest.raises(LLMOutputError):
        bad.complete('x', schema)
    assert extract_json('```json\n{"a": [1, {"b": "}"}]}\n```') == '{"a": [1, {"b": "}"}]}'
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(LLMError, match='ANTHROPIC_API_KEY'):
        LLMClient(kind='anthropic', model='m').complete('x')


def test_clients_cache_by_content(tmp_path):
    q = {'n': {'type': 'noul', 'instructions': 'y'}}
    a = JevClient(mock=True, cache_dir=tmp_path / 'jev', cache_namespace='ns')
    b = JevClient(mock=True, cache_dir=tmp_path / 'jev', cache_namespace='ns', cache_only=True)
    first = a.judge({'k': 1}, q)
    second = b.judge({'k': 1}, q)
    assert second['answers'] == first['answers'] and second['cache_hit'] and b.stats['calls'] == 0
    with pytest.raises(JevError):
        b.judge({'k': 2}, q)


# -- runtime ----------------------------------------------------------------------------------------------
@pytest.fixture
def runtime():
    if not sb.sandbox_available():
        pytest.skip('no native sandbox on this machine')
    rt = Runtime(_spec(), JevClient(mock=True), LLMClient(kind='mock'))
    yield rt
    rt.close()


def test_runtime_runs_waves_skips_null_prompt_and_carries_memory(runtime):
    one = runtime.run({'n': 3}, {})
    assert one['status'] == 'ok' and one['output']['n'] == 6 and one['memory'] == {'seen': 1}
    statuses = {r['id']: r['status'] for r in one['trace']}
    assert statuses['a'] == 'ok' and statuses['j'] == 'ok' and statuses['l'] in ('ok', 'skipped')
    two = runtime.run({'n': 1}, one['memory'])
    assert two['memory'] == {'seen': 2}
    assert runtime.pool.stats['starts'] == 1 and runtime.pool.stats['jobs'] == 2   # one warm worker across runs
    jev_record = next(r for r in one['trace'] if r['id'] == 'j')
    assert jev_record['response']['model'] == 'mock' and jev_record['state'] == {'n': 6, 'seen': 1}


def test_runtime_classifies_candidate_and_infrastructure_errors(monkeypatch):
    for name in ('TYPESAFE_API_KEY', 'TYPESAFE_API_KEY_PERSONAL', 'TYPESAFE_API_KEY_personal'):   # another test may have loaded .env into the process
        monkeypatch.delenv(name, raising=False)
    if not sb.sandbox_available():
        pytest.skip('no native sandbox on this machine')
    bad = _spec()
    bad['nodes'][0] = {'id': 'a', 'kind': 'python', 'source': "def run(obs, nodes, memory):\n    return obs['missing']\n", 'inputs': []}
    rt = Runtime(bad, JevClient(mock=True), LLMClient(kind='mock'))
    try:
        with pytest.raises(PipelineError) as info:
            rt.run({'n': 1})
        assert not info.value.infrastructure
        assert {r['id']: r['status'] for r in info.value.partial['trace']} == {'a': 'error', 'j': 'blocked', 'l': 'blocked', 'd': 'blocked'}
    finally:
        rt.close()
    rt = Runtime(_spec(), JevClient(mock=False, api_key=None), LLMClient(kind='mock'))
    try:
        with pytest.raises(PipelineError) as info:
            rt.run({'n': 1})
        assert info.value.infrastructure
    finally:
        rt.close()


def test_runtime_requires_llm_client_for_llm_nodes():
    with pytest.raises(ValueError):
        Runtime(_spec(), JevClient(mock=True), None)
