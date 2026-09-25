"""The harness spec: a JSON graph of python, jev, llm and expression nodes.

    {"format": "mixedharness/1", "name": str, "jev_model": str, "llm_model": str?,
     "nodes": [node, ...], "output": expr, "memory": expr?}

    python:     {"id", "kind": "python", "source": str, "inputs": [node ids]}      inputs are explicit
    expression: {"id", "kind": "expression", "expression": expr}                    inputs inferred
    jev:        {"id", "kind": "jev", "state": expr, "questions": map | "questions_expression": expr}
    llm:        {"id", "kind": "llm", "prompt": expr, "schema": json-schema?, "max_tokens": int?}

Dependencies are inferred from literal `nodes['id']` references in expressions and, for python nodes, taken
from `inputs`. A python node sees only its declared inputs. The graph must be acyclic. `output` and
`memory` run after all nodes; `memory` must produce an object.

Jev questions: {"id": {"type": "choice"|"score"|"noul", "instructions": str, "criteria": ...}} where choice
criteria map option ids to descriptions, score criteria are an ordered list of 2 to 10 level descriptions,
and noul takes no criteria.
"""
import copy
import hashlib
import json

from . import expressions
from .schema import check_schema, minimal_instance, validate as validate_json

FORMAT = 'mixedharness/1'
KINDS = ('python', 'expression', 'jev', 'llm')
QUESTION_TYPES = ('choice', 'score', 'noul')
MAX_NODES = 64
MAX_QUESTIONS = 64
MAX_OPTIONS = 100


class SpecError(ValueError):
    pass


def _ident(value, what):
    if not isinstance(value, str) or not value or len(value) > 64 or not value.replace('_', '').replace('-', '').isalnum() or value[0] == '_':
        raise SpecError(f'{what} must be a short identifier (letters, digits, - and _), got {value!r}')
    return value


def validate_questions(questions):
    if not isinstance(questions, dict) or not questions:
        raise SpecError('questions must be a nonempty object')
    if len(questions) > MAX_QUESTIONS:
        raise SpecError('too many questions')
    for qid, question in questions.items():
        _ident(qid, 'question id')
        if not isinstance(question, dict):
            raise SpecError(f'question {qid} must be an object')
        if set(question) - {'type', 'instructions', 'criteria'}:
            raise SpecError(f'question {qid}: unknown fields {sorted(set(question) - {"type", "instructions", "criteria"})}')
        kind = question.get('type')
        if kind not in QUESTION_TYPES:
            raise SpecError(f'question {qid}: type must be one of {QUESTION_TYPES}')
        if not isinstance(question.get('instructions'), str) or not question['instructions'].strip():
            raise SpecError(f'question {qid}: instructions must be a nonempty string')
        criteria = question.get('criteria')
        if kind == 'noul':
            if criteria is not None:
                raise SpecError(f'question {qid}: noul takes no criteria')
        elif kind == 'choice':
            if not isinstance(criteria, dict) or not criteria or len(criteria) > MAX_OPTIONS:
                raise SpecError(f'question {qid}: choice criteria must map 1 to {MAX_OPTIONS} option ids to descriptions')
            for option, description in criteria.items():
                if not isinstance(option, str) or not option or not isinstance(description, str):
                    raise SpecError(f'question {qid}: option ids and descriptions must be strings')
        else:
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10 or not all(isinstance(c, str) for c in criteria):
                raise SpecError(f'question {qid}: score criteria must be a list of 2 to 10 level descriptions')
    return questions


def validate_spec(spec):
    """Return a deep copy of a valid spec; raise SpecError otherwise."""
    if not isinstance(spec, dict):
        raise SpecError('spec must be an object')
    allowed = {'format', 'name', 'jev_model', 'llm_model', 'nodes', 'output', 'memory'}
    if set(spec) - allowed:
        raise SpecError(f'unknown spec fields {sorted(set(spec) - allowed)}')
    if spec.get('format') != FORMAT:
        raise SpecError(f'format must be {FORMAT!r}')
    if not isinstance(spec.get('name'), str) or not spec['name'].strip():
        raise SpecError('name must be a nonempty string')
    if not isinstance(spec.get('jev_model'), str) or not spec['jev_model']:
        raise SpecError('jev_model must be a nonempty string')
    if 'llm_model' in spec and (not isinstance(spec['llm_model'], str) or not spec['llm_model']):
        raise SpecError('llm_model must be a nonempty string when present')
    nodes = spec.get('nodes')
    if not isinstance(nodes, list) or not nodes or len(nodes) > MAX_NODES:
        raise SpecError(f'nodes must be a list of 1 to {MAX_NODES} nodes')
    ids = []
    for node in nodes:
        if not isinstance(node, dict):
            raise SpecError('each node must be an object')
        node_id = _ident(node.get('id'), 'node id')
        if node_id in ids:
            raise SpecError(f'duplicate node id {node_id}')
        ids.append(node_id)
    has_llm = False
    for node in nodes:
        kind = node.get('kind')
        if kind not in KINDS:
            raise SpecError(f"node {node['id']}: kind must be one of {KINDS}")
        fields = set(node)
        if kind == 'python':
            if fields - {'id', 'kind', 'source', 'inputs'}:
                raise SpecError(f"node {node['id']}: unknown fields")
            from .sandbox import validate_source
            validate_source(node.get('source'))
            inputs = node.get('inputs')
            if not isinstance(inputs, list) or any(i not in ids or i == node['id'] for i in inputs) or len(set(inputs)) != len(inputs):
                raise SpecError(f"node {node['id']}: inputs must list distinct other node ids")
        elif kind == 'expression':
            if fields - {'id', 'kind', 'expression'}:
                raise SpecError(f"node {node['id']}: unknown fields")
            _expr(node.get('expression'), node['id'], 'expression', ids)
        elif kind == 'jev':
            if fields - {'id', 'kind', 'state', 'questions', 'questions_expression'}:
                raise SpecError(f"node {node['id']}: unknown fields")
            _expr(node.get('state'), node['id'], 'state', ids)
            if ('questions' in node) == ('questions_expression' in node):
                raise SpecError(f"node {node['id']}: give exactly one of questions or questions_expression")
            if 'questions' in node:
                validate_questions(node['questions'])
            else:
                _expr(node['questions_expression'], node['id'], 'questions_expression', ids)
        else:
            has_llm = True
            if fields - {'id', 'kind', 'prompt', 'schema', 'max_tokens'}:
                raise SpecError(f"node {node['id']}: unknown fields")
            _expr(node.get('prompt'), node['id'], 'prompt', ids)
            if 'schema' in node:
                check_schema(node['schema'])
                validate_json(minimal_instance(node['schema']), node['schema'])
            if 'max_tokens' in node and (isinstance(node['max_tokens'], bool) or not isinstance(node['max_tokens'], int) or node['max_tokens'] < 1):
                raise SpecError(f"node {node['id']}: max_tokens must be a positive integer")
    if has_llm and 'llm_model' not in spec:
        raise SpecError('llm nodes require llm_model')
    _expr(spec.get('output'), None, 'output', ids)
    if 'memory' in spec:
        _expr(spec['memory'], None, 'memory', ids)
    dependencies(spec)  # raises on cycles
    return copy.deepcopy(spec)


def _expr(expression, node_id, field, ids):
    try:
        refs = expressions.references(expression)
    except expressions.ExpressionError as exc:
        where = f'node {node_id} {field}' if node_id else field
        raise SpecError(f'{where}: {exc}') from None
    for ref in refs:
        if ref not in ids or ref == node_id:
            where = f'node {node_id} {field}' if node_id else field
            raise SpecError(f'{where}: references unknown or own node {ref!r}')
    return refs


def node_inputs(node):
    """Node ids this node reads: explicit for python, inferred from expressions otherwise."""
    kind = node['kind']
    if kind == 'python':
        return list(node['inputs'])
    if kind == 'expression':
        return expressions.references(node['expression'])
    if kind == 'jev':
        refs = expressions.references(node['state'])
        if 'questions_expression' in node:
            refs += [r for r in expressions.references(node['questions_expression']) if r not in refs]
        return refs
    return expressions.references(node['prompt'])


def dependencies(spec):
    """{node_id: [input ids]} in spec order; raises SpecError on a cycle."""
    deps = {node['id']: node_inputs(node) for node in spec['nodes']}
    state = {}

    def visit(node_id, stack):
        if state.get(node_id) == 'done':
            return
        if state.get(node_id) == 'active':
            raise SpecError('dependency cycle: ' + ' -> '.join(stack + [node_id]))
        state[node_id] = 'active'
        for dep in deps[node_id]:
            visit(dep, stack + [node_id])
        state[node_id] = 'done'

    for node_id in deps:
        visit(node_id, [])
    return deps


def spec_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def authoring_schema():
    """The JSON Schema handed to an authoring or reflection model. Describes the format, not the task."""
    expr = {'type': 'string', 'description': "Restricted expression over obs, nodes['<id>'] and memory: literals, indexing, arithmetic, comparisons, conditionals, and the functions len min max sum abs round mean median std last get clip column text sorted any all."}
    question = {'type': 'object', 'required': ['type', 'instructions'], 'additionalProperties': False,
                'properties': {'type': {'type': 'string', 'enum': list(QUESTION_TYPES)},
                               'instructions': {'type': 'string'},
                               'criteria': {'description': 'choice: {option_id: description}; score: ordered list of 2-10 level descriptions; noul: omitted'}}}
    return {
        'title': 'Mixed Harness spec', 'type': 'object', 'additionalProperties': False,
        'required': ['format', 'name', 'jev_model', 'nodes', 'output'],
        'properties': {
            'format': {'type': 'string', 'const': FORMAT},
            'name': {'type': 'string'},
            'jev_model': {'type': 'string', 'description': 'fixed; never change'},
            'llm_model': {'type': 'string', 'description': 'fixed; required when any llm node exists; never change'},
            'output': expr, 'memory': dict(expr, description='optional; must evaluate to an object carried into the next run as memory'),
            'nodes': {'type': 'array', 'items': {'oneOf': [
                {'type': 'object', 'required': ['id', 'kind', 'source', 'inputs'], 'additionalProperties': False,
                 'properties': {'id': {'type': 'string'}, 'kind': {'const': 'python'},
                                'source': {'type': 'string', 'description': 'Defines run(obs, nodes, memory) plus helper functions. No imports, no classes, no attribute writes; builtins limited to a safe subset; returns a JSON value.'},
                                'inputs': {'type': 'array', 'items': {'type': 'string'}, 'description': 'node ids visible as nodes[...] inside run'}}},
                {'type': 'object', 'required': ['id', 'kind', 'expression'], 'additionalProperties': False,
                 'properties': {'id': {'type': 'string'}, 'kind': {'const': 'expression'}, 'expression': expr}},
                {'type': 'object', 'required': ['id', 'kind', 'state'], 'additionalProperties': False,
                 'properties': {'id': {'type': 'string'}, 'kind': {'const': 'jev'}, 'state': dict(expr, description='the JSON state Jev judges'),
                                'questions': {'type': 'object', 'additionalProperties': question},
                                'questions_expression': dict(expr, description='expression producing the question map at runtime; exactly one of questions / questions_expression')}},
                {'type': 'object', 'required': ['id', 'kind', 'prompt'], 'additionalProperties': False,
                 'properties': {'id': {'type': 'string'}, 'kind': {'const': 'llm'},
                                'prompt': dict(expr, description='expression producing the prompt string, or null to skip the node this run (no call, output null)'),
                                'schema': {'type': 'object', 'description': 'optional JSON Schema subset the reply must satisfy; with it the node output is the parsed JSON, without it the text'},
                                'max_tokens': {'type': 'integer', 'minimum': 1}}}]}},
        },
        'description': ('A harness is a directed acyclic graph. Independent nodes run in parallel. python nodes compute facts and '
                        'decisions; jev nodes ask bounded judgment questions (several per request); llm nodes generate text or '
                        'structured JSON and may be skipped by returning a null prompt. Node outputs are read as nodes[id].'),
    }
