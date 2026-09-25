"""Run one decision through a spec: nodes in dependency waves, each recorded in full.

    result = Runtime(spec, jev, llm).run(obs, memory)
    result = {'output': json, 'memory': dict, 'memory_before': dict, 'nodes': {id: output},
              'trace': [record, ...], 'elapsed_ms': float, 'status': 'ok'}

A record has `id`, `kind`, `status` (ok / skipped / error / blocked), `inputs`, `elapsed_ms`, `output`, and
what the node saw and got back: `source` and `worker_ms` for python, `expression` for expression nodes,
`state`, `questions` and `response` for jev, `prompt`, `schema` and `response` for llm. Nodes that fail
raise `PipelineError` after the wave finishes, with the partial result attached and `infrastructure=True`
when any cause was a provider or sandbox outage rather than the candidate's own fault.
"""
import concurrent.futures
import copy
import hashlib
import time

from . import expressions
from .jev import JevError
from .llm import LLMError, LLMOutputError
from .sandbox import NodeError, SandboxUnavailable, SourceError, WorkerPool, validate_source
from .schema import SchemaError
from .spec import dependencies, validate_questions, validate_spec

INFRASTRUCTURE = (JevError, LLMError, SandboxUnavailable)


class PipelineError(RuntimeError):
    def __init__(self, message, *, partial=None, causes=None):
        super().__init__(message)
        self.partial = partial or {}
        self.causes = list(causes or [])
        self.infrastructure = any(isinstance(c, INFRASTRUCTURE) for c in self.causes)


def _ms(start):
    return (time.perf_counter() - start) * 1000


class Runtime:
    def __init__(self, spec, jev, llm=None, *, max_workers=8, pool=None):
        self.spec = validate_spec(spec)
        self.jev = jev
        self.llm = llm
        self.nodes = {node['id']: node for node in self.spec['nodes']}
        self.deps = dependencies(self.spec)
        self.order = [node['id'] for node in self.spec['nodes']]
        if any(node['kind'] == 'llm' for node in self.spec['nodes']) and llm is None:
            raise ValueError('the spec has llm nodes; pass an LLM client')
        if any(node['kind'] == 'agent' for node in self.spec['nodes']) and llm is None:
            raise ValueError('the spec has agent nodes; pass an LLM client')
        self.max_workers = max(1, int(max_workers))
        self.pool = pool or (WorkerPool(size=self.max_workers) if any(n['kind'] == 'python' for n in self.spec['nodes']) else None)

    def close(self):
        if self.pool is not None:
            self.pool.close()

    # -- execution -------------------------------------------------------------------------------------
    def run(self, obs, memory=None):
        start = time.perf_counter()
        expressions.check_json(obs)
        memory_before = copy.deepcopy(memory) if isinstance(memory, dict) else {}
        expressions.check_json(memory_before)
        outputs, records, errors = {}, {}, {}
        pending = set(self.order)
        done = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while pending:
                ready = [n for n in self.order if n in pending and all(d in done for d in self.deps[n])]
                blocked = [n for n in self.order if n in pending and any(d in errors for d in self.deps[n])]
                for node_id in blocked:
                    failed = next(d for d in self.deps[node_id] if d in errors)
                    records[node_id] = self._record(self.nodes[node_id], status='blocked', error=f'input {failed!r} failed')
                    pending.discard(node_id); done.add(node_id); errors[node_id] = None
                ready = [n for n in ready if n not in errors]
                if not ready:
                    if blocked:
                        continue   # blocking may have unblocked-by-failure further nodes; re-scan
                    if pending:
                        raise PipelineError('no runnable node; the graph is inconsistent', partial={'trace': list(records.values())})
                    break
                futures = {executor.submit(self._execute, self.nodes[n], outputs, obs, memory_before): n for n in ready}
                for future, node_id in futures.items():
                    record, value, error = future.result()
                    records[node_id] = record
                    pending.discard(node_id); done.add(node_id)
                    if error is not None:
                        errors[node_id] = error
                    else:
                        outputs[node_id] = value
        trace = [records[n] for n in self.order]
        causes = [e for e in errors.values() if e is not None]
        if causes:
            partial = {'output': None, 'memory': memory_before, 'memory_before': memory_before,
                       'nodes': outputs, 'trace': trace, 'elapsed_ms': _ms(start), 'status': 'error'}
            failed = [n for n in self.order if n in errors and errors[n] is not None]
            raise PipelineError(f'node(s) failed: {", ".join(failed)}: {causes[0]}', partial=partial, causes=causes)
        context = {'obs': obs, 'nodes': outputs, 'memory': memory_before}
        try:
            output = expressions.evaluate(self.spec['output'], context)
            memory_after = expressions.evaluate(self.spec['memory'], context) if 'memory' in self.spec else copy.deepcopy(memory_before)
            if not isinstance(memory_after, dict):
                raise expressions.ExpressionError('memory must evaluate to an object')
        except expressions.ExpressionError as exc:
            partial = {'output': None, 'memory': memory_before, 'memory_before': memory_before,
                       'nodes': outputs, 'trace': trace, 'elapsed_ms': _ms(start), 'status': 'error'}
            raise PipelineError(f'finalisation failed: {exc}', partial=partial, causes=[exc])
        return {'output': output, 'memory': memory_after, 'memory_before': memory_before, 'nodes': outputs,
                'trace': trace, 'elapsed_ms': _ms(start), 'status': 'ok'}

    def _record(self, node, **fields):
        record = {'id': node['id'], 'kind': node['kind'], 'status': 'ok', 'inputs': list(self.deps[node['id']]),
                  'elapsed_ms': 0.0, 'output': None}
        if node['kind'] == 'python':
            record['source_sha256'] = hashlib.sha256(node['source'].encode('utf-8')).hexdigest()
        elif node['kind'] == 'expression':
            record['expression'] = node['expression']
        elif node['kind'] == 'jev':
            record.update(state=None, questions=copy.deepcopy(node.get('questions')), response=None)
        elif node['kind'] == 'agent':
            record.update(instructions=node['instructions'], context=None, response=None)
        else:
            record.update(prompt=None, schema=copy.deepcopy(node.get('schema')), response=None)
        record.update(fields)
        return record

    def _execute(self, node, outputs, obs, memory):
        """Run one node; returns (record, value, error). Never raises for candidate-level failures."""
        start = time.perf_counter()
        record = self._record(node)
        visible = {dep: copy.deepcopy(outputs[dep]) for dep in self.deps[node['id']]}
        context = {'obs': copy.deepcopy(obs), 'nodes': visible, 'memory': copy.deepcopy(memory)}
        value, error = None, None
        try:
            kind = node['kind']
            if kind == 'python':
                validate_source(node['source'])
                value, worker_ms = self.pool.run(node['source'], context)
                record['worker_ms'] = worker_ms
            elif kind == 'expression':
                value = expressions.evaluate(node['expression'], context)
            elif kind == 'jev':
                state = expressions.evaluate(node['state'], context)
                questions = node['questions'] if 'questions' in node else expressions.evaluate(node['questions_expression'], context)
                validate_questions(questions)
                record['state'] = copy.deepcopy(state)
                record['questions'] = copy.deepcopy(questions)
                response = self.jev.judge(state, questions, model=self.spec['jev_model'])
                record['response'] = copy.deepcopy(response)
                value = response['answers']
            elif kind == 'agent':
                agent_context = expressions.evaluate(node['context'], context)
                if agent_context is None:
                    record['status'] = 'skipped'
                else:
                    record['context'] = copy.deepcopy(agent_context)
                    response = self.llm.turn(node['instructions'], agent_context, model=self.spec['llm_model'], max_tokens=node.get('max_tokens'))
                    record['response'] = copy.deepcopy(response)
                    value = {k: response.get(k) for k in ('kind', 'name', 'arguments', 'text')}
            else:
                prompt = expressions.evaluate(node['prompt'], context)
                if prompt is None:
                    record['status'] = 'skipped'
                elif not isinstance(prompt, str) or not prompt:
                    raise expressions.ExpressionError('prompt must evaluate to a nonempty string or null')
                else:
                    record['prompt'] = prompt
                    response = self.llm.complete(prompt, node.get('schema'), model=self.spec['llm_model'], max_tokens=node.get('max_tokens'))
                    record['response'] = copy.deepcopy(response)
                    value = response['value']
            expressions.check_json(value)
            record['output'] = copy.deepcopy(value)
        except (NodeError, SourceError, expressions.ExpressionError, LLMOutputError, SchemaError, ValueError,
                JevError, LLMError, SandboxUnavailable) as exc:
            error = exc
            record['status'] = 'error'
            record['error'] = f'{type(exc).__name__}: {str(exc)[:500]}'
            record['infrastructure'] = isinstance(exc, INFRASTRUCTURE)
        record['elapsed_ms'] = _ms(start)
        return record, value, error
