"""A tau2-bench HalfDuplexAgent that runs one Runtime call per turn.

Each ledger entry also carries the guard node's routing record (cluster, closed flag, Jev's route pick, the
realised path and why, agreement, confidence) when the spec has one, and `steps()` renders the ledger as
step-store rows.

The pipeline sees a bounded, parsed view of the conversation plus the tool schemas and policy, and returns
either a tool call or a message. Memory produced by the pipeline is carried to the next turn. Every turn is
recorded in a ledger with realised counters: python node runs, Jev calls, LLM calls, tokens, elapsed time,
and the model ids the providers actually echoed. Nothing in the ledger is a configured value.

Infrastructure errors (Jev outage, LLM provider failure, sandbox unavailable) propagate and abort the
simulation; candidate-level errors (an invalid action, a node bug) become a fallback message and are counted.
"""
import copy
import json
import time
import uuid

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage

from mixedharness.runtime import PipelineError, Runtime
MAX_CONVERSATION = 40
MAX_RESULT_CHARS = 6000
FALLBACK_TEXT = "I'm sorry, I did not catch that. Could you tell me a bit more about what you would like to do?"
JEV_LIST_PRICE_PER_INPUT_TOKEN = 0.042 / 1_000_000  # docs.typesafe.ai/models, jev-1.13, used only when the response carries no cost


class PipelineAgentState:
    def __init__(self, history):
        self.history = list(history)      # parsed message dicts, oldest first
        self.memory = {}
        self.turn = 0
        self.consecutive_errors = 0


def tool_schema(tool):
    schema = tool.openai_schema
    schema = schema() if callable(schema) else schema
    fn = schema.get('function', schema)
    params = fn.get('parameters') or {}
    props = params.get('properties') or {}
    return {'description': (tool.short_desc or tool.long_desc or fn.get('description') or '')[:400],
            'required': list(params.get('required') or []),
            'params': {name: {'type': str(p.get('type', 'string')), 'description': str(p.get('description', ''))[:200]}
                       for name, p in props.items()}}


def _parse_result(content):
    if content is None:
        return None
    text = content if isinstance(content, str) else json.dumps(content)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS]
        try:
            return json.loads(text)
        except ValueError:
            return text + '…'
    try:
        return json.loads(text)
    except ValueError:
        return text


def parse_message(message, tool_names_by_id):
    """tau2 message -> observation dict. Returns a list because MultiToolMessage carries several."""
    if isinstance(message, MultiToolMessage):
        out = []
        for m in message.tool_messages:
            out.extend(parse_message(m, tool_names_by_id))
        return out
    if isinstance(message, ToolMessage):
        name, arguments = tool_names_by_id.get(message.id, (None, None))
        return [{'role': 'tool', 'text': None, 'tool': name, 'arguments': arguments,
                 'result': _parse_result(message.content), 'error': bool(message.error)}]
    if isinstance(message, UserMessage):
        return [{'role': 'user', 'text': message.content or '', 'tool': None, 'arguments': None, 'result': None, 'error': False}]
    if isinstance(message, AssistantMessage):
        if message.tool_calls:
            call = message.tool_calls[0]
            return [{'role': 'assistant', 'text': None, 'tool': call.name, 'arguments': dict(call.arguments or {}),
                     'result': None, 'error': False}]
        return [{'role': 'assistant', 'text': message.content or '', 'tool': None, 'arguments': None, 'result': None, 'error': False}]
    return []


class PipelineAgent(HalfDuplexAgent[PipelineAgentState]):
    def __init__(self, tools, domain_policy, spec, jev, llm=None, *, label='pipeline', max_workers=8, pool=None):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.spec = spec
        self.jev = jev
        self.llm = llm
        self.label = label
        # A shared pool keeps the sandbox process count flat across simulations: one pool per arm, not one per agent.
        self.runtime = Runtime(spec, jev, llm, max_workers=max_workers, pool=pool)
        self.owns_pool = pool is None
        self.tool_schemas = {tool.name: tool_schema(tool) for tool in tools}
        self.tool_names_by_id = {}
        self.ledger = []
        self.counters = {'turns': 0, 'jev_calls': 0, 'llm_calls': 0, 'llm_skipped': 0, 'python_runs': 0,
                         'jev_input_tokens': 0, 'llm_input_tokens': 0, 'llm_output_tokens': 0,
                         'candidate_errors': 0, 'invalid_actions': 0, 'elapsed_ms': 0.0, 'cost_usd': 0.0,
                         'unpriced_calls': 0}
        self.actual_models = {'jev': set(), 'llm': set()}

    def close(self):
        if self.owns_pool:
            self.runtime.close()

    # -- tau2 contract ----------------------------------------------------------------------------
    def get_init_state(self, message_history=None):
        history = []
        for message in message_history or []:
            history.extend(parse_message(message, self.tool_names_by_id))
        return PipelineAgentState(history)

    def generate_next_message(self, message, state):
        state.turn += 1
        incoming = parse_message(message, self.tool_names_by_id)
        incoming_obs = incoming[-1] if incoming else {'role': 'user', 'text': '', 'tool': None, 'arguments': None, 'result': None, 'error': False}
        obs = {'turn': state.turn, 'policy': self.domain_policy, 'tools': copy.deepcopy(self.tool_schemas),
               'conversation': list(state.history[-MAX_CONVERSATION:]), 'incoming': incoming_obs}
        state.history.extend(incoming)
        started = time.perf_counter()
        decision, error = None, None
        try:
            decision = self.runtime.run(obs, memory=state.memory)
        except PipelineError as exc:
            if exc.infrastructure:
                raise
            error = exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        action, invalid = self._action_from(decision, error)
        entry = self._ledger_entry(state, obs, decision, error, action, invalid, elapsed_ms)
        self.ledger.append(entry)
        if decision is not None and error is None:
            state.memory = decision['memory'] if isinstance(decision.get('memory'), dict) else state.memory
            state.consecutive_errors = 0 if not invalid else state.consecutive_errors + 1
        else:
            state.consecutive_errors += 1
        reply = self._assistant_message(action, entry)
        state.history.extend(parse_message(reply, self.tool_names_by_id))
        return reply, state

    # -- helpers ----------------------------------------------------------------------------------
    def _action_from(self, decision, error):
        if error is not None or decision is None:
            return {'kind': 'message', 'text': FALLBACK_TEXT}, 'pipeline_error'
        action = decision.get('output')
        if not isinstance(action, dict):
            return {'kind': 'message', 'text': FALLBACK_TEXT}, 'output_not_object'
        kind = action.get('kind')
        if kind == 'message':
            text = action.get('text')
            if not isinstance(text, str) or not text.strip():
                return {'kind': 'message', 'text': FALLBACK_TEXT}, 'empty_message'
            return {'kind': 'message', 'text': text.strip()}, None
        if kind == 'tool':
            name = action.get('name')
            schema = self.tool_schemas.get(name)
            if schema is None:
                return {'kind': 'message', 'text': FALLBACK_TEXT}, 'unknown_tool'
            arguments = action.get('arguments')
            if not isinstance(arguments, dict):
                return {'kind': 'message', 'text': FALLBACK_TEXT}, 'arguments_not_object'
            unknown = [k for k in arguments if k not in schema['params']]
            missing = [k for k in schema['required'] if k not in arguments]
            if unknown or missing:
                return {'kind': 'message', 'text': FALLBACK_TEXT}, f'bad_arguments unknown={unknown} missing={missing}'
            return {'kind': 'tool', 'name': name, 'arguments': arguments}, None
        return {'kind': 'message', 'text': FALLBACK_TEXT}, 'unknown_kind'

    def _ledger_entry(self, state, obs, decision, error, action, invalid, elapsed_ms):
        jev_calls = llm_calls = llm_skipped = python_runs = 0
        jev_tokens = llm_in = llm_out = 0
        cost, unpriced = 0.0, 0
        model_ms = sandbox_ms = 0.0   # model-only latency vs sandboxed-python process overhead
        trace = (decision or {}).get('trace') if decision else None
        if trace is None and error is not None and isinstance(error.partial, dict):
            trace = error.partial.get('trace')
        for node in trace or []:
            kind = node.get('kind')
            elapsed = float(node.get('elapsed_ms') or 0.0)
            if kind == 'python':
                sandbox_ms += elapsed
            elif kind in ('jev', 'llm', 'agent'):
                model_ms += elapsed
            if kind == 'python' and node.get('status') == 'ok':
                python_runs += 1
            elif kind == 'jev' and node.get('response'):
                jev_calls += 1
                resp = node['response']
                usage = resp.get('usage') or {}
                tokens = usage.get('inputTokens', usage.get('input_tokens', 0)) or 0
                jev_tokens += tokens
                if resp.get('model'):
                    self.actual_models['jev'].add(str(resp['model']))
                node_cost = resp.get('cost_usd')
                if node_cost is None:
                    gateway = (resp.get('gateway_metadata') or {}).get('provider_metadata') or {}
                    node_cost = (gateway.get('gateway') or {}).get('cost')
                if node_cost is not None:
                    cost += float(node_cost)
                elif tokens:
                    cost += tokens * JEV_LIST_PRICE_PER_INPUT_TOKEN
                else:
                    unpriced += 1
            elif kind in ('llm', 'agent'):
                if node.get('status') == 'skipped':
                    llm_skipped += 1
                elif node.get('soft'):
                    llm_calls += 1   # the call happened and was paid for; the reply was unusable
                    c_soft = self.counters
                    c_soft['llm_soft_errors'] = c_soft.get('llm_soft_errors', 0) + 1
                elif node.get('response'):
                    llm_calls += 1
                    resp = node['response']
                    usage = resp.get('usage') or {}
                    llm_in += usage.get('input_tokens', 0) or 0
                    llm_out += usage.get('output_tokens', 0) or 0
                    if resp.get('model'):
                        self.actual_models['llm'].add(str(resp['model']))
                    if resp.get('cost_usd') is not None:
                        cost += float(resp['cost_usd'])
                    else:
                        unpriced += 1
        c = self.counters
        c['turns'] += 1; c['jev_calls'] += jev_calls; c['llm_calls'] += llm_calls; c['llm_skipped'] += llm_skipped
        c['python_runs'] += python_runs; c['jev_input_tokens'] += jev_tokens; c['llm_input_tokens'] += llm_in
        c['llm_output_tokens'] += llm_out; c['elapsed_ms'] += elapsed_ms; c['cost_usd'] += cost; c['unpriced_calls'] += unpriced
        c['model_ms'] = c.get('model_ms', 0.0) + model_ms; c['sandbox_ms'] = c.get('sandbox_ms', 0.0) + sandbox_ms
        if invalid:
            c['invalid_actions'] += 1
        if error is not None:
            c['candidate_errors'] += 1
        chosen, routing, guard = None, None, None
        for node in trace or []:
            if node.get('id') in ('decision', 'guard') and isinstance(node.get('output'), dict):
                chosen = node['output'].get('chosen', chosen)
            if node.get('id') == 'guard' and isinstance(node.get('output'), dict):
                routing = node['output'].get('routing')
                guard = node['output'].get('guard')
        return {'turn': state.turn, 'elapsed_ms': elapsed_ms, 'model_ms': model_ms, 'sandbox_ms': sandbox_ms,
                'routing': routing, 'guard': guard,
                'jev_calls': jev_calls, 'llm_calls': llm_calls,
                'llm_skipped': llm_skipped, 'python_runs': python_runs, 'jev_input_tokens': jev_tokens,
                'llm_input_tokens': llm_in, 'llm_output_tokens': llm_out, 'cost_usd': cost, 'unpriced_calls': unpriced,
                'chosen': chosen, 'action': action, 'invalid': invalid,
                'error': str(error) if error is not None else None,
                'observation': obs, 'decision': decision if decision is not None else (error.partial if error is not None else None)}

    def _assistant_message(self, action, entry):
        usage = {'jev_input_tokens': entry['jev_input_tokens'], 'llm_input_tokens': entry['llm_input_tokens'],
                 'llm_output_tokens': entry['llm_output_tokens'], 'jev_calls': entry['jev_calls'], 'llm_calls': entry['llm_calls']}
        common = {'role': 'assistant', 'cost': entry['cost_usd'] if entry['unpriced_calls'] == 0 else None,
                  'usage': usage, 'generation_time_seconds': entry['elapsed_ms'] / 1000.0}
        if action['kind'] == 'tool':
            call_id = 'call_' + uuid.uuid4().hex[:12]
            self.tool_names_by_id[call_id] = (action['name'], dict(action['arguments']))
            return AssistantMessage(content=None, tool_calls=[ToolCall(id=call_id, name=action['name'], arguments=dict(action['arguments']), requestor='assistant')], **common)
        return AssistantMessage(content=action['text'], tool_calls=None, **common)

    def steps(self, *, episode, arm, trial=0, reward=None):
        """Step-store rows for this agent's ledger (mixedharness.trace)."""
        from mixedharness.spec import spec_hash
        from mixedharness.trace import steps_from_ledger
        return steps_from_ledger(self.ledger, episode=episode, arm=arm, spec_hash=spec_hash(self.spec), trial=trial,
                                 harness_version=self.spec.get('name'), reward=reward)

    def summary(self):
        return {'label': self.label, 'counters': dict(self.counters),
                'actual_models': {k: sorted(v) for k, v in self.actual_models.items()},
                'spec_name': self.spec.get('name'), 'jev_model': self.spec.get('jev_model'), 'llm_model': self.spec.get('llm_model')}
