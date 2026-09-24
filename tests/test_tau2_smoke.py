"""Plumbing checks for the tau2 three-arm setup. Everything here runs on mock providers.

The tau2-dependent tests skip when tau2-bench is not importable (it lives in its own Python 3.12 venv,
see docs/tau2.md); the runtime rule they rely on is tested without tau2.
"""
import pytest

from auto_jev.llm import LLMClient
from auto_jev.providers import JevClient
from auto_jev.runtime import PipelineRuntime
from auto_jev.spec import validate_spec


def test_llm_node_with_null_prompt_is_skipped_without_a_call():
    spec = validate_spec({'version': 4, 'name': 'skip', 'jev_model': 'typesafe-ai/jev', 'llm_model': 'mock-llm',
                          'nodes': [{'id': 'gate', 'kind': 'expression', 'expression': "obs['want'] if obs['want'] else None"},
                                    {'id': 'draft', 'kind': 'llm', 'prompt_expression': "nodes['gate']"}],
                          'output': "nodes['draft']"})
    llm = LLMClient(kind='mock', model='mock-llm')
    runtime = PipelineRuntime(spec, JevClient(mock=True), llm)
    skipped = runtime.run({'want': None})
    assert skipped['output'] is None
    assert [n['status'] for n in skipped['trace'] if n['id'] == 'draft'] == ['skipped']
    assert llm.stats['calls'] == 0
    called = runtime.run({'want': 'say hi'})
    assert called['output'] == 'say hi'
    assert llm.stats['calls'] == 1


def test_seeds_validate_and_differ_only_in_llm_nodes():
    from examples.tau2 import seeds
    jev_only, harness = seeds.jev_only_seed(), seeds.harness_seed()
    assert [n['kind'] for n in jev_only['nodes']] == ['python', 'jev', 'python']
    assert [n['kind'] for n in harness['nodes']] == ['python', 'jev', 'python', 'llm', 'llm', 'python']
    assert 'llm_model' not in jev_only and harness['llm_model'] == 'claude-sonnet-5'


def _tau2():
    try:
        from examples.tau2 import adapter
        adapter.ensure_tau2()
        import tau2  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tau2(), reason='tau2-bench not importable in this interpreter')
def test_pipeline_agent_turns_are_valid_tau2_messages_and_counted():
    from tau2.data_model.message import ToolMessage, UserMessage
    from tau2.runner import build_environment
    from examples.tau2 import seeds
    from examples.tau2.agent import PipelineAgent
    env = build_environment('retail')
    for name, llm in (('jev_only', None), ('harness', LLMClient(kind='mock', model='mock-llm'))):
        spec = seeds.SEEDS[name]()
        agent = PipelineAgent(env.get_tools(), env.get_policy(), spec, JevClient(mock=True), llm, label=name)
        state = agent.get_init_state([])
        reply, state = agent.generate_next_message(UserMessage(role='user', content='Hi, I am Yusuf Rossi, zip 19122, I want to exchange an item in #W2378156.'), state)
        reply.validate()
        assert reply.is_tool_call() or reply.content
        if reply.is_tool_call():
            call = reply.tool_calls[0]
            assert call.name == 'find_user_id_by_name_zip'
            assert call.arguments == {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}
            result = env.get_response(call)
            assert isinstance(result, ToolMessage)
            reply2, state = agent.generate_next_message(result, state)
            reply2.validate()
            assert state.memory.get('user_id') == 'yusuf_rossi_9620'
        counters = agent.counters
        assert counters['turns'] >= 1 and counters['jev_calls'] == counters['turns']
        assert counters['llm_calls'] == 0  # mock Jev never picks reply:llm; harness llm nodes are skipped, not called
        if name == 'harness':
            assert counters['llm_skipped'] == 2 * counters['turns']
        assert counters['invalid_actions'] == 0 and counters['candidate_errors'] == 0
