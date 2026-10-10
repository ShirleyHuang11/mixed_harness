"""Routing layer, step table, macro-action mining, and the tau2 routed harness on scripted providers.

The tau2 turn tests build the observation by hand (a small tool schema and a policy string) and drive the routed
seed through the Runtime with a scripted Jev (route pick, action pick, paraphrase) and a scripted LLM (menu,
agent turn), so every path and the guard's write gate are exercised without tau2-bench installed.
"""
import json

import pytest

from examples.tau2 import seeds, seeds_splits
from mixedharness import sandbox as sb
from mixedharness.jev import JevClient
from mixedharness.llm import LLMClient
from mixedharness.mining import mine_macro_actions
from mixedharness.router import NEVER, PATHS, agreement, compact_policy, fit_policy, fit_tau, merge_policy, policy_source_literal, route_metrics
from mixedharness.runtime import Runtime
from mixedharness.spec import FORMAT, validate_spec
from mixedharness.trace import steps_from_ledger

needs_sandbox = pytest.mark.skipif(not sb.sandbox_available(), reason='no native sandbox on this machine')


# -- router: signals and the per-cluster policy -----------------------------------------------------------
def test_agreement_reads_the_action_and_its_paraphrase():
    ans = {'action': {'choice': 'a'}, 'action_alt': {'choice': 'a'}}
    assert agreement(ans) == (True, {'action': 'a', 'action_alt': 'a'})
    assert agreement({'action': {'choice': 'a'}, 'action_alt': {'choice': 'b'}})[0] is False
    assert agreement({'action': {'choice': 'a'}})[0] is None


def _step(cluster, path, reward, cost=0.0, **extra):
    return {'cluster': cluster, 'path': path, 'reward': reward, 'cost_usd': cost, **extra}


def test_fit_policy_prefers_the_best_scored_path_only_with_enough_samples():
    steps = [_step('c1', 'closed', 1.0) for _ in range(10)] + [_step('c1', 'escalate', 1.0, cost=0.02) for _ in range(10)]
    steps += [_step('c2', 'escalate', 1.0, cost=0.02) for _ in range(10)] + [_step('c2', 'closed', 0.0) for _ in range(10)]
    steps += [_step('c3', 'closed', 1.0) for _ in range(3)]          # too few to prefer anything
    steps += [_step('c1', 'closed', None)]                            # unscored rows are ignored
    policy = fit_policy(steps, min_samples=8, epsilon=0.1)
    assert policy['clusters']['c1']['prefer'] == 'closed'            # same success, cheaper
    assert policy['clusters']['c2']['prefer'] == 'escalate'          # closed fails there
    assert policy['clusters']['c3']['prefer'] is None
    compact = compact_policy(policy)
    assert compact == {'epsilon': 0.1, 'prefer': {'c1': 'closed', 'c2': 'escalate'}, 'tau': {}, 'defaults': {}, 'splits': []}
    import ast
    assert ast.literal_eval(policy_source_literal(policy)) == compact
    assert ast.literal_eval(policy_source_literal(None)) == {'epsilon': 0.0, 'prefer': {}, 'tau': {}, 'defaults': {}, 'splits': []}


def test_route_metrics_and_mining_on_step_rows():
    rows = [{'episode': 'e1', 'trial': 0, 'turn': 1, 'path': 'closed', 'path_source': 'jev', 'agreement': True, 'closed': True, 'action_kind': 'tool', 'action_name': 'a'},
            {'episode': 'e1', 'trial': 0, 'turn': 2, 'path': 'escalate', 'path_source': 'jev', 'agreement': None, 'closed': False, 'action_kind': 'tool', 'action_name': 'x'},
            {'episode': 'e1', 'trial': 0, 'turn': 3, 'path': 'escalate', 'path_source': 'jev', 'agreement': None, 'closed': False, 'action_kind': 'tool', 'action_name': 'y'},
            {'episode': 'e2', 'trial': 0, 'turn': 1, 'path': 'narrow', 'path_source': 'fallthrough', 'agreement': False, 'closed': True, 'action_kind': 'tool', 'action_name': 'x', 'guard': 'write_gate'},
            {'episode': 'e2', 'trial': 0, 'turn': 2, 'path': 'escalate', 'path_source': 'jev', 'agreement': None, 'closed': False, 'action_kind': 'tool', 'action_name': 'y'},
            {'episode': 'e3', 'trial': 0, 'turn': 1, 'path': 'escalate', 'path_source': 'trajectory', 'agreement': None, 'closed': True, 'action_kind': 'tool', 'action_name': 'x'},
            {'episode': 'e3', 'trial': 0, 'turn': 2, 'path': 'escalate', 'path_source': 'jev', 'agreement': None, 'closed': False, 'action_kind': 'tool', 'action_name': 'y'}]
    m = route_metrics(rows)
    assert m['steps'] == 7 and m['jev_traffic_share'] == pytest.approx(1 / 7) and m['path_sources']['fallthrough'] == pytest.approx(1 / 7)
    assert m['agreement_rate'] == 0.5 and m['guard_interventions'] == 1
    mined = mine_macro_actions(rows, min_support=3, max_len=2)
    assert mined['llm_tool_steps'] == 6
    assert mined['candidates'][0]['sequence'] == ['x', 'y'] and mined['candidates'][0]['support'] == 3
    assert route_metrics([]) == {'steps': 0}


def test_steps_from_ledger_reads_the_guard_routing_record():
    ledger = [{'turn': 1, 'elapsed_ms': 12.0, 'model_ms': 4.0, 'jev_calls': 1, 'llm_calls': 0, 'cost_usd': 0.001, 'invalid': None, 'chosen': 'tool:x',
               'routing': {'cluster': 'anon|user|no_order|other', 'closed': True, 'needs': {}, 'path_jev': 'closed', 'path': 'closed', 'path_source': 'jev',
                           'agreement': True, 'confidence': 0.9, 'fits': 0.8},
               'guard': 'ok', 'action': {'kind': 'tool', 'name': 'x', 'arguments': {}}, 'decision': {'nodes': {}}}]
    rows = steps_from_ledger(ledger, episode='t1', arm='routed', spec_hash='abc', reward=1.0)
    assert rows[0]['path'] == 'closed' and rows[0]['cluster'] == 'anon|user|no_order|other' and rows[0]['reward'] == 1.0
    assert rows[0]['action_name'] == 'x' and rows[0]['guard'] == 'ok' and rows[0]['spec_hash'] == 'abc'


# -- runtime: a jev node with a null question map is skipped ----------------------------------------------
@needs_sandbox
def test_jev_node_with_null_questions_is_skipped_without_a_call():
    spec = validate_spec({'format': FORMAT, 'name': 'skip', 'jev_model': 'jev-1.13.0',
                          'nodes': [{'id': 'q', 'kind': 'expression', 'expression': "{'a': {'type': 'noul', 'instructions': 'x'}} if obs['ask'] else None"},
                                    {'id': 'j', 'kind': 'jev', 'state': "{'k': 1}", 'questions_expression': "nodes['q']"}],
                          'output': "nodes['j']"})
    jev = JevClient(mock=True)
    rt = Runtime(spec, jev)
    try:
        skipped = rt.run({'ask': False})
        assert skipped['output'] is None and [n['status'] for n in skipped['trace'] if n['id'] == 'j'] == ['skipped'] and jev.stats['calls'] == 0
        asked = rt.run({'ask': True})
        assert asked['output']['a']['type'] == 'noul' and jev.stats['calls'] == 1
    finally:
        rt.close()


# -- the tau2 routed harness, one turn at a time on scripted providers ------------------------------------
TOOLS = {
    'find_user_id_by_name_zip': {'description': 'Find user id by name and zip.', 'required': ['first_name', 'last_name', 'zip'],
                                 'params': {'first_name': {'type': 'string', 'description': ''}, 'last_name': {'type': 'string', 'description': ''}, 'zip': {'type': 'string', 'description': ''}}},
    'find_user_id_by_email': {'description': 'Find user id by email.', 'required': ['email'], 'params': {'email': {'type': 'string', 'description': ''}}},
    'modify_user_address': {'description': 'Change the default address.', 'required': ['user_id', 'address1', 'address2', 'city', 'state', 'country', 'zip'],
                            'params': {k: {'type': 'string', 'description': ''} for k in ('user_id', 'address1', 'address2', 'city', 'state', 'country', 'zip')}},
    'get_user_details': {'description': 'User profile.', 'required': ['user_id'], 'params': {'user_id': {'type': 'string', 'description': ''}}},
    'get_order_details': {'description': 'Order.', 'required': ['order_id'], 'params': {'order_id': {'type': 'string', 'description': ''}}},
    'cancel_pending_order': {'description': 'Cancel.', 'required': ['order_id', 'reason'], 'params': {'order_id': {'type': 'string', 'description': ''}, 'reason': {'type': 'string', 'description': ''}}},
    'transfer_to_human_agents': {'description': 'Transfer.', 'required': ['summary'], 'params': {'summary': {'type': 'string', 'description': ''}}},
    'return_delivered_order_items': {'description': 'Return.', 'required': ['order_id', 'item_ids', 'payment_method_id'],
                                     'params': {'order_id': {'type': 'string', 'description': ''}, 'item_ids': {'type': 'array', 'description': ''}, 'payment_method_id': {'type': 'string', 'description': ''}}},
    'exchange_delivered_order_items': {'description': 'Exchange.', 'required': ['order_id', 'item_ids', 'new_item_ids', 'payment_method_id'],
                                       'params': {'order_id': {'type': 'string', 'description': ''}, 'item_ids': {'type': 'array', 'description': ''}, 'new_item_ids': {'type': 'array', 'description': ''}, 'payment_method_id': {'type': 'string', 'description': ''}}},
}
HELLO = 'Hi, I am Yusuf Rossi, zip 19122, I want to cancel order #W2378156.'
ORDER = {'order_id': '#W2378156', 'status': 'delivered', 'items': [{'name': 'Keyboard', 'item_id': '1', 'price': 100, 'product_id': 'p1', 'options': {}}],
         'address': {'city': 'Philadelphia'}, 'payment_history': [{'payment_method_id': 'credit_card_9513926'}]}
AUTH = {'user_id': 'yusuf_rossi_9620', 'user': {'user_id': 'yusuf_rossi_9620', 'orders': ['#W2378156', '#W0000001'], 'payment_methods': {}}, 'orders': {'#W2378156': ORDER}}


def _obs(text, conversation=(), incoming=None, turn=1):
    msg = {'role': 'user', 'text': text, 'tool': None, 'arguments': None, 'result': None, 'error': False}
    return {'turn': turn, 'policy': 'Authenticate first. Confirm before any write.', 'tools': json.loads(json.dumps(TOOLS)),
            'conversation': list(conversation), 'incoming': incoming or msg}


def _jev(route, action=None, alt=None, confidence=0.95):
    """A scripted Jev: fixed route pick; action and paraphrase picks (default: the first option, agreeing)."""
    def responder(state, questions):
        out = {}
        if 'route' in questions:
            out['route'] = {'choice': route, 'probabilities': {p: (0.9 if p == route else 0.05) for p in PATHS}}
        if 'action' in questions:
            options = list(questions['action']['criteria'])
            a = action or options[0]
            out['action'] = {'choice': a, 'probabilities': {o: (confidence if o == a else 0.0) for o in options}}
            b = alt or a
            out['action_alt'] = {'choice': b, 'probabilities': {o: (0.95 if o == b else 0.0) for o in options}}
        if 'pick' in questions:
            first = list(questions['pick']['criteria'])[0]
            out['pick'] = {'choice': first, 'probabilities': {first: 1.0}}
        return out
    return JevClient(mock=True, mock_responder=responder)


def _llm(menu=None, agent=None):
    def responder(first, second):
        if isinstance(first, str):        # complete(prompt, schema): the menu
            return json.dumps(menu or {'options': [{'id': 'a', 'kind': 'message', 'text': 'Menu reply.', 'when': 'always'}]})
        return agent or {'kind': 'message', 'name': None, 'arguments': None, 'text': 'Agent reply.'}   # turn(context, tools)
    return LLMClient(kind='mock', model='mock-llm', mock_responder=responder)


def _run(spec, jev, llm, obs, memory=None):
    rt = Runtime(spec, jev, llm)
    try:
        return rt.run(obs, memory or {})
    finally:
        rt.close()


def _statuses(result):
    return {r['id']: r['status'] for r in result['trace']}


@needs_sandbox
def test_structural_step_runs_no_model_at_all():
    llm = _llm()
    result = _run(seeds_splits.routed_seed('mock-llm'), _jev('closed', 'tool:find_user_id_by_name_zip'), llm, _obs(HELLO))
    guard = result['nodes']['guard']
    assert result['output'] == {'kind': 'tool', 'name': 'find_user_id_by_name_zip', 'arguments': {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}}
    assert guard['routing']['path'] == 'closed' and guard['routing']['path_source'] == 'structural'   # one name pair and a zip: code decides
    assert guard['routing']['closed'] is True and guard['guard'] == 'ok'
    s = _statuses(result)
    assert s['judge'] == 'skipped' and s['agent'] == 'skipped' and llm.stats['calls'] == 0
    assert result['nodes']['compile']['cluster'] == 'anon|user|no_order|cancel'


RETURN = 'I want to return the keyboard from order #W2378156.'


@needs_sandbox
def test_jev_pick_is_accepted_by_its_own_probability():
    llm = _llm()
    jev = _jev('closed', 'tool:return_delivered_order_items', confidence=0.97)
    result = _run(seeds.routed_seed('mock-llm'), jev, llm, _obs(RETURN, turn=5), AUTH)
    routing = result['nodes']['guard']['routing']
    assert routing['path'] == 'closed' and routing['path_source'] == 'jev' and routing['confidence'] == 0.97 and jev.stats['calls'] == 1
    assert llm.stats['calls'] == 0


@needs_sandbox
def test_escalate_path_write_gate_then_confirmed_write():
    write = {'kind': 'tool', 'name': 'cancel_pending_order', 'arguments': {'order_id': '#W2378156', 'reason': 'no longer needed'}, 'text': None}
    spec = seeds.routed_seed('mock-llm')
    first = _run(spec, _jev('closed', confidence=0.3), _llm(agent=write), _obs(RETURN, turn=5), AUTH)
    guard = first['nodes']['guard']
    assert guard['routing']['path'] == 'escalate' and guard['guard'] == 'write_gate'
    assert first['output']['kind'] == 'message' and 'confirm' in first['output']['text'].lower()
    assert first['memory']['pending'] == {'tool': 'cancel_pending_order', 'arguments': write['arguments']}
    conversation = [_obs(RETURN)['incoming'], {'role': 'assistant', 'text': first['output']['text'], 'tool': None, 'arguments': None, 'result': None, 'error': False}]
    second = _run(spec, _jev('closed', confidence=0.3), _llm(agent=write), _obs('Yes, please go ahead.', conversation, turn=6), first['memory'])
    assert second['output'] == {'kind': 'tool', 'name': 'cancel_pending_order', 'arguments': write['arguments']}
    assert second['nodes']['guard']['guard'] == 'ok'


@needs_sandbox
def test_invalid_agent_action_falls_back_to_the_closed_decision():
    bad = {'kind': 'tool', 'name': 'no_such_tool', 'arguments': {}, 'text': None}
    result = _run(seeds.routed_seed('mock-llm'), _jev('closed', confidence=0.3), _llm(agent=bad), _obs(RETURN, turn=5), AUTH)
    guard = result['nodes']['guard']
    assert guard['guard'] == 'unknown_tool' and guard['routing']['path'] == 'closed' and guard['routing']['path_source'] == 'guard'
    assert result['output']['kind'] == 'message'


@needs_sandbox
def test_trajectory_escalation_after_a_repeated_action():
    spec = seeds.routed_seed('mock-llm')
    memory = {'same_reply_streak': seeds.STUCK_REPLIES}
    result = _run(spec, _jev('closed', 'tool:find_user_id_by_name_zip'), _llm(), _obs(HELLO), memory)
    routing = result['nodes']['guard']['routing']
    assert routing['path'] == 'escalate' and routing['path_source'] == 'trajectory'


@needs_sandbox
def test_shadow_share_flips_a_jev_decided_step():
    shadow = seeds_splits.routed_seed('mock-llm', policy={'epsilon': 1.0, 'clusters': {}})
    result = _run(shadow, _jev('closed', 'tool:return_delivered_order_items', confidence=0.97), _llm(), _obs(RETURN, turn=5), AUTH)
    routing = result['nodes']['guard']['routing']
    assert routing['path_source'] == 'shadow' and routing['shadow_of'] == 'closed' and routing['path'] == 'escalate'
    structural = _run(shadow, _jev('closed'), _llm(), _obs(HELLO))
    assert structural['nodes']['guard']['routing']['path_source'] == 'structural'   # code-decided steps are never shadowed


@needs_sandbox
def test_jev_only_seed_stays_closed_whatever_the_confidence():
    result = _run(seeds.jev_only_seed(), _jev('closed', 'tool:return_delivered_order_items', confidence=0.3), None, _obs(RETURN, turn=5), AUTH)
    routing = result['nodes']['guard']['routing']
    assert routing['path'] == 'closed' and routing['path_source'] == 'forced' and routing['confidence'] == 0.3
    assert [n['kind'] for n in seeds.jev_only_seed()['nodes']] == ['python', 'jev', 'python', 'python', 'python']


@needs_sandbox
def test_open_request_is_flagged_open_and_overrides_a_closed_pick():
    """An authenticated customer asking a question the catalog does not address: fillable options, open request."""
    llm = _llm()
    memory = {'user_id': 'yusuf_rossi_9620', 'user': {'user_id': 'yusuf_rossi_9620', 'orders': ['#W2378156'], 'payment_methods': {'credit_card_1': {'source': 'credit_card', 'brand': 'visa', 'last_four': '1234'}}}}
    result = _run(seeds.routed_seed('mock-llm'), _jev('closed', 'reply:ask_order'), llm, _obs('Can you explain why refunds take so long?', turn=3), memory)
    cx = result['nodes']['compile']
    assert cx['closed'] is False and cx['needs'] == {} and cx['facts']['request_covered'] is False
    routing = result['nodes']['guard']['routing']
    assert routing['path'] == 'escalate' and routing['path_source'] == 'code_override'   # narrow is off by default
    covered = _run(seeds.routed_seed('mock-llm'), _jev('closed', 'tool:get_order_details:#W2378156'), llm, _obs('Please cancel order #W2378156.', turn=3), memory)
    assert covered['nodes']['compile']['closed'] is True


@needs_sandbox
def test_refusal_clears_the_pending_write_and_cancel_that_is_a_yes():
    spec = seeds.jev_only_seed()
    memory = {'user_id': 'u', 'user': {'user_id': 'u', 'orders': ['#W1'], 'payment_methods': {}}, 'orders': {'#W1': {'order_id': '#W1', 'status': 'pending', 'items': []}},
              'pending': {'tool': 'cancel_pending_order', 'arguments': {'order_id': '#W1', 'reason': 'no longer needed'}}}
    said_no = _run(spec, _jev('closed', 'tool:cancel_pending_order'), None, _obs('No, do not cancel it.', turn=4), memory)
    assert said_no['output']['kind'] == 'message' and 'will not proceed' in said_no['output']['text'] and said_no['memory']['pending'] is None
    from examples.tau2.seeds import COMMON_SOURCE
    namespace = {}
    exec(COMMON_SOURCE, namespace)   # the helper as the sandbox sees it
    assert namespace['yes_no']('Yes, go ahead and cancel that order') == 'yes' and namespace['yes_no']('No, wait') == 'no'


@needs_sandbox
def test_low_confidence_closed_pick_escalates():
    llm = _llm()
    result = _run(seeds.routed_seed('mock-llm'), _jev('closed', 'tool:return_delivered_order_items', confidence=0.5), llm, _obs(RETURN, turn=5), AUTH)
    routing = result['nodes']['guard']['routing']
    assert routing['path'] == 'escalate' and routing['path_source'] == 'low_confidence' and routing['confidence'] == 0.5
    lenient = _run(seeds.routed_seed('mock-llm', tau=0.4), _jev('closed', 'tool:return_delivered_order_items', confidence=0.5), llm, _obs(RETURN, turn=5), AUTH)
    assert lenient['nodes']['guard']['routing']['path'] == 'closed'


@needs_sandbox
def test_a_loaded_order_becomes_the_focus_and_is_not_fetched_again():
    result = _run(seeds.routed_seed('mock-llm'), _jev('closed'), _llm(), _obs('I want to return the keyboard.', turn=5), AUTH)
    cx = result['nodes']['compile']
    assert cx['facts']['focus_order'] == '#W2378156' and 'tool:get_order_details:#W2378156' not in cx['options']
    refetch = {'kind': 'tool', 'name': 'get_order_details', 'arguments': {'order_id': '#W2378156'}, 'text': None}
    again = _run(seeds.routed_seed('mock-llm'), _jev('closed', confidence=0.3), _llm(agent=refetch), _obs('What is in that order?', turn=6), AUTH)
    guard = again['nodes']['guard']
    assert guard['guard'] == 'memory_answer' and again['output']['kind'] == 'message' and 'Keyboard' in again['output']['text']


@needs_sandbox
def test_write_gate_matches_tool_and_order_not_exact_arguments():
    first = {'kind': 'tool', 'name': 'cancel_pending_order', 'arguments': {'order_id': '#W2378156', 'reason': 'no longer needed'}, 'text': None}
    spec = seeds.routed_seed('mock-llm')
    one = _run(spec, _jev('closed', confidence=0.3), _llm(agent=first), _obs('Please cancel order #W2378156.'), {})
    assert one['nodes']['guard']['guard'] == 'write_gate'
    reworded = {'kind': 'tool', 'name': 'cancel_pending_order', 'arguments': {'order_id': '#W2378156', 'reason': 'ordered by mistake'}, 'text': None}
    conv = [_obs('Please cancel order #W2378156.')['incoming'], {'role': 'assistant', 'text': one['output']['text'], 'tool': None, 'arguments': None, 'result': None, 'error': False}]
    two = _run(spec, _jev('closed', confidence=0.3), _llm(agent=reworded), _obs('Yes, go ahead.', conv, turn=2), one['memory'])
    assert two['output']['kind'] == 'tool' and two['nodes']['guard']['guard'] == 'ok'


@needs_sandbox
def test_repeated_identical_tool_call_counts_as_stuck():
    spec = seeds.jev_only_seed()
    memory = dict(AUTH, orders={})
    first = _run(spec, _jev('closed', 'tool:get_order_details:#W2378156'), None, _obs('Tell me about order #W2378156.', turn=3), memory)
    assert first['output']['name'] == 'get_order_details' and first['memory']['same_reply_streak'] == 0
    second = _run(spec, _jev('closed', 'tool:get_order_details:#W2378156'), None, _obs('Tell me about order #W2378156.', turn=4), first['memory'])
    assert second['memory']['same_reply_streak'] == 1 and second['memory']['last_signature'].startswith('tool:get_order_details')


def test_fit_tau_picks_the_smallest_reliable_threshold_per_cluster():
    def rep(cluster, conf, ok):
        return {'label_closed': True, 'cluster': cluster, 'jev_confidence': conf, 'jev_correct': ok}
    rows = [rep('easy', 0.7, True) for _ in range(20)]                                   # reliable at low confidence
    rows += [rep('hard', 0.95, False) for _ in range(10)] + [rep('hard', 0.995, True) for _ in range(10)]   # only >=0.99 is reliable
    rows += [rep('hopeless', 0.999, False) for _ in range(10)] + [rep('hopeless', 0.999, True) for _ in range(2)]
    rows += [rep('thin', 0.99, True) for _ in range(3)]
    fitted = fit_tau(rows, target=0.9, min_n=10)
    c = fitted['clusters']
    assert c['easy']['tau'] == 0.6 and c['hard']['tau'] == 0.99 and c['hopeless']['tau'] == NEVER and 'thin' not in c
    policy = merge_policy(None, fitted)
    compact = compact_policy(policy)
    assert compact['tau'] == {'easy': 0.6, 'hard': 0.99, 'hopeless': NEVER} and compact['defaults'] == {}
    import ast
    assert ast.literal_eval(policy_source_literal(policy))['tau']['hard'] == 0.99


@needs_sandbox
def test_route_uses_the_cluster_threshold_over_the_global_one():
    llm = _llm()
    cluster = 'auth|user|delivered|return'
    lax = {'epsilon': 0.0, 'prefer': {}, 'tau': {cluster: 0.6}}
    result = _run(seeds_splits.routed_seed('mock-llm', policy=lax), _jev('closed', 'tool:return_delivered_order_items', confidence=0.7), llm, _obs(RETURN, turn=5), AUTH)
    routing = result['nodes']['guard']['routing']
    assert routing['cluster'] == cluster and routing['tau'] == 0.9 and routing['path'] == 'escalate'   # never laxer than the global tau
    strict = {'epsilon': 0.0, 'prefer': {}, 'tau': {cluster: NEVER}}
    result = _run(seeds_splits.routed_seed('mock-llm', policy=strict), _jev('closed', 'tool:return_delivered_order_items', confidence=1.0), llm, _obs(RETURN, turn=5), AUTH)
    assert result['nodes']['guard']['routing']['path'] == 'escalate'


@needs_sandbox
def test_write_gate_accepts_a_yes_after_the_llm_asked_in_prose():
    write = {'kind': 'tool', 'name': 'cancel_pending_order', 'arguments': {'order_id': '#W2378156', 'reason': 'no longer needed'}, 'text': None}
    conv = [_obs('Please cancel order #W2378156.')['incoming'],
            {'role': 'assistant', 'text': 'I can cancel order #W2378156 (reason: no longer needed). Shall I proceed?', 'tool': None, 'arguments': None, 'result': None, 'error': False}]
    result = _run(seeds.routed_seed('mock-llm'), _jev('closed', confidence=0.3), _llm(agent=write), _obs('Yes please.', conv, turn=3), {})
    assert result['output']['kind'] == 'tool' and result['nodes']['guard']['guard'] == 'ok'
    unrelated = [_obs('Please cancel order #W2378156.')['incoming'],
                 {'role': 'assistant', 'text': 'Your order contains a keyboard and a thermostat.', 'tool': None, 'arguments': None, 'result': None, 'error': False}]
    result = _run(seeds.routed_seed('mock-llm'), _jev('closed', confidence=0.3), _llm(agent=write), _obs('Yes.', unrelated, turn=3), {})
    assert result['nodes']['guard']['guard'] == 'write_gate'


def test_fit_from_steps_learns_code_rows_and_tightenings():
    from mixedharness.router import fit_from_steps
    def st(cluster, chosen, reward, source='jev'):
        return {'cluster': cluster, 'path': 'closed', 'path_source': source, 'chosen': chosen, 'reward': reward}
    steps = [st('a', 'tool:x', 1.0) for _ in range(12)]                                # always the same pick, winning
    steps += [st('b', 'tool:y', 0.0) for _ in range(8)] + [st('b', 'tool:y', 1.0) for _ in range(4)]   # same pick but losing
    steps += [st('c', 'tool:p', 1.0) for _ in range(6)] + [st('c', 'tool:q', 1.0) for _ in range(6)]   # split picks
    steps += [{'cluster': 'd', 'path': 'escalate', 'path_source': 'low_confidence', 'chosen': None, 'reward': 1.0} for _ in range(20)]
    learned = fit_from_steps(steps, min_n=10)
    assert learned['defaults'] == {'a': 'tool:x'} and learned['tighten'] == ['b']


CANCEL_NO_ID = 'Hi, I want to cancel an order I placed last week.'


@needs_sandbox
def test_learned_split_row_lets_code_take_the_step():
    """A split row (cluster + state features -> action class) fitted offline fires in the compiler: no model is called."""
    policy = {'epsilon': 0.0, 'prefer': {}, 'tau': {}, 'defaults': {},
              'splits': [{'cluster': 'anon|user|no_order|cancel', 'split': {'user_loaded': False, 'n_mentioned': 0}, 'action': 'reply:ask_identity'}]}
    llm = _llm()
    result = _run(seeds_splits.routed_seed('mock-llm', policy=policy), _jev('closed', 'reply:ask_identity', confidence=0.2), llm, _obs(CANCEL_NO_ID))
    routing = result['nodes']['guard']['routing']
    assert routing['path'] == 'closed' and routing['path_source'] == 'learned' and result['output']['kind'] == 'message'
    assert _statuses(result)['judge'] == 'skipped' and llm.stats['calls'] == 0
    assert routing['features']['user_loaded'] is False and routing['features']['n_mentioned'] == 0 and routing['features']['last'] == 'user'
    # without the row the same step is open to Jev / the LLM
    plain = _run(seeds_splits.routed_seed('mock-llm'), _jev('closed', 'reply:ask_identity', confidence=0.2), _llm(), _obs(CANCEL_NO_ID))
    assert plain['nodes']['guard']['routing']['path_source'] in ('low_confidence', 'code_override')


@needs_sandbox
def test_offline_features_match_runtime_features():
    """The learner recomputes state_features from a ledger entry; it must equal what the compiler recorded."""
    from scripts.learn_splits import features_of
    result = _run(seeds_splits.routed_seed('mock-llm'), _jev('closed', 'tool:return_delivered_order_items', confidence=0.97), _llm(), _obs(RETURN, turn=5), AUTH)
    routing = result['nodes']['guard']['routing']
    entry = {'observation': _obs(RETURN, turn=5), 'decision': {'memory': result['memory']}, 'routing': {k: v for k, v in routing.items() if k != 'features'}}
    f, leaky = features_of(entry)
    assert leaky
    for k, v in routing['features'].items():
        if k not in ('pending', 'n_completed'):
            assert f[k] == v, (k, f[k], v)
    f2, leaky2 = features_of({'routing': routing})
    assert not leaky2 and f2 == routing['features']


ADDRESS = 'Please update my default address to 123 Main St, Springfield, IL, USA, 62701.'


@needs_sandbox
def test_address_change_is_listed_and_left_to_the_llm():
    """An address change is in the catalog (it was missing: Jev then picked the nearest listed reply and closed the
    conversation); its fields are not fillable by code, so the step is open and the LLM takes it."""
    result = _run(seeds.routed_seed('mock-llm'), _jev('closed', 'reply:closing', confidence=0.99), _llm(), _obs(ADDRESS, turn=5), AUTH)
    cx = result['nodes']['compile']
    assert 'tool:modify_user_address' in cx['options'] and 'address1' in cx['needs']['tool:modify_user_address']
    assert cx['closed'] is False and cx['facts']['remaining_requests'] == ['modify_address']
    routing = result['nodes']['guard']['routing']
    assert routing['path'] == 'escalate' and routing['path_source'] == 'code_override'


@needs_sandbox
def test_closing_is_not_offered_while_a_request_remains():
    done = dict(AUTH, completed=[{'tool': 'cancel_pending_order', 'arguments': {'order_id': '#W0000001'}}])
    result = _run(seeds.routed_seed('mock-llm'), _jev('closed', confidence=0.3), _llm(), _obs('Thanks. I also want to exchange the keyboard for the black one.', turn=7), done)
    cx = result['nodes']['compile']
    assert cx['facts']['remaining_requests'] == ['exchange'] and 'reply:closing' not in cx['options']
    bye = _run(seeds.routed_seed('mock-llm'), _jev('closed', 'reply:closing', confidence=0.99), _llm(), _obs("That's all, thanks, bye!", turn=8), done)
    assert 'reply:closing' in bye['nodes']['compile']['options'] and bye['nodes']['compile']['facts']['remaining_requests'] == []
