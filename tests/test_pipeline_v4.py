import pytest

from auto_jev.flow import compile_flow
from auto_jev.llm import LLMClient
from auto_jev.runtime import PipelineExecutionError, PipelineRuntime
from auto_jev.spec import pipeline_schema, validate_spec
from tests.support import ScriptedJev

SCHEMA = {'type': 'object', 'required': ['answer'], 'properties': {'answer': {'type': 'string', 'enum': ['yes', 'no']}}}


def v4(**overrides):
    spec = {'version': 4, 'name': 'llm flow', 'jev_model': 'typesafe-ai/jev', 'llm_model': 'mock-llm',
            'nodes': [
                {'id': 'prompt', 'kind': 'expression', 'expression': "'Question: ' + obs['q']"},
                {'id': 'draft', 'kind': 'llm', 'prompt_expression': "nodes['prompt']", 'output_schema': SCHEMA},
                {'id': 'check', 'kind': 'jev', 'state': "{'draft': nodes['draft']}",
                 'questions': {'ok': {'type': 'noul', 'instructions': 'Is the draft acceptable?'}}},
            ],
            'output': "{'answer': nodes['draft']['answer'], 'ok': nodes['check']['ok']['noul']}"}
    spec.update(overrides)
    return spec


def test_v4_validates_and_infers_llm_dependencies():
    spec = validate_spec(v4())
    assert compile_flow(spec) == {'prompt': [], 'draft': ['prompt'], 'check': ['draft']}
    schema = pipeline_schema(version=4)
    assert schema['properties']['version'] == {'const': 4}
    assert any(kind['properties'].get('kind') == {'const': 'llm'} for kind in schema['properties']['nodes']['items']['oneOf'])


@pytest.mark.parametrize('mutate', [
    lambda s: s.pop('llm_model'),
    lambda s: s['nodes'][1].pop('prompt_expression'),
    lambda s: s['nodes'][1].update(extra=1),
    lambda s: s['nodes'][1].update(output_schema={'type': 'object', 'patternProperties': {}}),
    lambda s: s.update(version=3),
])
def test_invalid_v4_specs_are_rejected(mutate):
    spec = v4(); mutate(spec)
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_v4_runs_llm_then_jev_and_records_prompt_and_response():
    llm = LLMClient(kind='mock', mock_responder=lambda prompt, schema: '{"answer": "yes"}')
    runtime = PipelineRuntime(v4(), ScriptedJev({'ok': 0.9}), llm)
    result = runtime.run({'q': 'go?'})
    assert result['output'] == {'answer': 'yes', 'ok': 0.9}
    record = result['trace'][1]
    assert record['kind'] == 'llm' and record['prompt'] == 'Question: go?'
    assert record['output_schema'] == SCHEMA and record['response']['value'] == {'answer': 'yes'}
    assert llm.stats['calls'] == 1


def test_llm_node_schema_failure_is_a_node_error():
    llm = LLMClient(kind='mock', mock_responder=lambda prompt, schema: '{"answer": "maybe"}')
    runtime = PipelineRuntime(v4(), ScriptedJev(), llm)
    with pytest.raises(PipelineExecutionError) as caught:
        runtime.run({'q': 'go?'})
    trace = caught.value.partial_result['trace']
    assert trace[1]['status'] == 'error' and 'schema' in trace[1]['error']
    assert trace[2]['status'] == 'blocked'


def test_llm_nodes_require_a_client():
    with pytest.raises(ValueError, match='llm'):
        PipelineRuntime(v4(), ScriptedJev())


def test_v3_specs_are_untouched_by_v4_fields():
    with pytest.raises(ValueError):
        validate_spec({**v4(), 'version': 3})
