"""Plumbing checks for the tau2 four-arm setup. Everything here runs on mock providers.

The tau2-dependent tests skip when tau2-bench is not importable (it lives in its own Python 3.12 venv,
see docs/guides/tau2.md); the runtime rule they rely on is tested without tau2.
"""
import pytest

from mixedharness.jev import JevClient
from mixedharness.llm import LLMClient
from mixedharness.runtime import Runtime
from mixedharness.spec import FORMAT, validate_spec


def test_llm_node_with_null_prompt_is_skipped_without_a_call():
    spec = validate_spec({'format': FORMAT, 'name': 'skip', 'jev_model': 'jev-1.13.0', 'llm_model': 'mock-llm',
                          'nodes': [{'id': 'gate', 'kind': 'expression', 'expression': "obs['want'] if obs['want'] else None"},
                                    {'id': 'draft', 'kind': 'llm', 'prompt': "nodes['gate']"}],
                          'output': "nodes['draft']"})
    llm = LLMClient(kind='mock', model='mock-llm')
    runtime = Runtime(spec, JevClient(mock=True), llm)
    skipped = runtime.run({'want': None})
    assert skipped['output'] is None
    assert [n['status'] for n in skipped['trace'] if n['id'] == 'draft'] == ['skipped']
    assert llm.stats['calls'] == 0
    called = runtime.run({'want': 'say hi'})
    assert called['output'] == 'say hi'
    assert llm.stats['calls'] == 1


def test_seeds_validate_and_the_controls_are_the_routed_graph_minus_the_llm_paths():
    from examples.tau2 import seeds
    jev_only, routed, llm_only = seeds.jev_only_seed(), seeds.routed_seed(), seeds.llm_only_seed()
    assert [n['id'] for n in jev_only['nodes']] == ['compile', 'judge', 'route', 'decision', 'guard']
    assert [n['id'] for n in routed['nodes']] == ['compile', 'judge', 'route', 'decision', 'menu', 'menu_q', 'menu_pick', 'agent', 'guard']   # the menu path is off by default and skipped
    assert [n['kind'] for n in llm_only['nodes']] == ['python', 'agent']
    assert 'llm_model' not in jev_only and routed['llm_model'] == 'claude-sonnet-5'
    assert jev_only['nodes'][0]['source'] == routed['nodes'][0]['source']   # the same compiler


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
    for name, llm in (('jev_only', None), ('routed', LLMClient(kind='mock', model='mock-llm'))):
        spec = seeds.jev_only_seed() if name == 'jev_only' else seeds.routed_seed('mock-llm')
        agent = PipelineAgent(env.get_tools(), env.get_policy(), spec, JevClient(mock=True), llm, label=name)
        state = agent.get_init_state([])
        reply, state = agent.generate_next_message(UserMessage(role='user', content='Hi, I am Yusuf Rossi, zip 19122, I want to exchange an item in #W2378156.'), state)
        reply.validate()
        assert reply.is_tool_call() or reply.content
        if reply.is_tool_call() and reply.tool_calls[0].name == 'find_user_id_by_name_zip':
            call = reply.tool_calls[0]
            assert call.arguments == {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}
            result = env.get_response(call)
            assert isinstance(result, ToolMessage)
            reply2, state = agent.generate_next_message(result, state)
            reply2.validate()
            assert state.memory.get('user_id') == 'yusuf_rossi_9620'
        counters = agent.counters
        assert counters['turns'] >= 1 and counters['jev_calls'] >= counters['turns']
        if name == 'jev_only':
            assert counters['llm_calls'] == 0 and counters['llm_skipped'] == 0
        rows = agent.steps(episode='smoke', arm=name)
        assert len(rows) == counters['turns'] and rows[0]['path'] in ('closed', 'narrow', 'escalate') and rows[0]['cluster']
        assert counters['invalid_actions'] == 0 and counters['candidate_errors'] == 0
