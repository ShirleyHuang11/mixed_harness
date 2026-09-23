"""Route a task before authoring: code probes evidence, Jev judges the route, templates seed each route."""
import copy
import json
import statistics

from .spec import _json, validate_questions, validate_spec

ROUTES = ('settled', 'open_ended', 'hybrid')
TARGET_OF = {'jev': 'settled', 'llm': 'open_ended', 'hybrid': 'hybrid', 'code': 'settled'}
ACTION_KINDS = ('labels', 'catalog', 'open', 'declared_unknown')
EVALUATOR_KINDS = ('exact_match', 'environment', 'code_check', 'reward_model', 'judge')
EXACT_EVALUATORS = ('exact_match', 'environment', 'code_check')
SHORT_TEXT_TOKENS = 12
EXAMPLE_LIMIT = 5
EXAMPLE_CHARS = 300

DEFAULT_TRIAGE_QUESTIONS = {
    'route': {
        'type': 'choice',
        'instructions': ('Using the probe evidence and examples, judge which decision form fits this task. '
                         'Evidence fields are computed by code from the task description; examples are representative.'),
        'criteria': {
            'settled': 'Every decision picks from a finite, enumerable set (labels or a per-step legal action list); '
                       'the output can be checked exactly by code; the judgment is choose, score or yes/no.',
            'open_ended': 'The output is free text, code or an unbounded structure that must be generated, '
                          'not selected; no small candidate set exists before the model writes it.',
            'hybrid': 'The output is open, but a few candidates can be generated first and then the best one '
                      'selected by comparison; quality is judged by comparing options.',
        },
    },
    'code_computes_facts': {'type': 'noul', 'instructions': 'Can deterministic code compute most facts needed for each decision, leaving only a bounded judgment to a model?'},
    'needs_multistep': {'type': 'noul', 'instructions': 'Does a single decision require multi-step reasoning or tool use before an answer can be given?'},
}


def validate_task(task):
    _json(task)
    if not isinstance(task, dict):
        raise ValueError('task must be an object')
    for key in ('name', 'objective', 'observation_fields', 'action_space', 'evaluator', 'decisions_per_episode', 'examples'):
        if key not in task:
            raise ValueError(f'task requires {key}')
    if not isinstance(task['examples'], list) or not task['examples']:
        raise ValueError('task examples must be a nonempty list')
    for example in task['examples']:
        if not isinstance(example, dict) or 'observation' not in example or 'expected' not in example:
            raise ValueError('each example needs observation and expected')
    action = task['action_space']
    if not isinstance(action, dict) or action.get('kind') not in ACTION_KINDS:
        raise ValueError('action_space.kind must be one of ' + ', '.join(ACTION_KINDS))
    if action['kind'] == 'labels' and (not isinstance(action.get('labels'), list) or not action['labels']):
        raise ValueError('labels action space needs a nonempty labels list')
    if action['kind'] == 'catalog' and not isinstance(action.get('catalog_field'), str):
        raise ValueError('catalog action space needs catalog_field')
    if not isinstance(task['evaluator'], dict) or task['evaluator'].get('kind') not in EVALUATOR_KINDS:
        raise ValueError('evaluator.kind must be one of ' + ', '.join(EVALUATOR_KINDS))
    if isinstance(task['decisions_per_episode'], bool) or not isinstance(task['decisions_per_episode'], int) or task['decisions_per_episode'] < 1:
        raise ValueError('decisions_per_episode must be a positive integer')
    if 'output_schema' in task and not isinstance(task['output_schema'], dict):
        raise ValueError('output_schema must be an object')
    return copy.deepcopy(task)


def _token_count(value):
    return len(str(value).split()) if isinstance(value, str) else len(json.dumps(value))


def probe(task):
    """Deterministic evidence from the task description; no model calls."""
    task = validate_task(task)
    action = task['action_space']
    labels = action.get('labels') if action['kind'] == 'labels' else None
    field = action.get('catalog_field')
    expected = [example['expected'] for example in task['examples']]
    if labels is not None and all(value in labels for value in expected):
        expected_kind = 'label'
    elif all(isinstance(value, str) for value in expected):
        median = statistics.median(_token_count(value) for value in expected)
        expected_kind = 'short_text' if median <= SHORT_TEXT_TOKENS else 'long_text'
    else:
        expected_kind = 'structured'
    return {
        'declared_action_kind': action['kind'],
        'label_count': len(labels) if labels is not None else None,
        'catalog_present_in_examples': bool(field) and all(
            isinstance(example['observation'].get(field), list) and example['observation'][field] for example in task['examples']),
        'exact_validator': task['evaluator']['kind'] in EXACT_EVALUATORS,
        'expected_kind': expected_kind,
        'median_expected_tokens': statistics.median(_token_count(value) for value in expected),
        'decisions_per_episode': task['decisions_per_episode'],
        'latency_budget_ms': task.get('latency_budget_ms'),
        'example_count': len(task['examples']),
    }


def _clip(value):
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= EXAMPLE_CHARS else text[:EXAMPLE_CHARS] + '…'


def triage(task, jev, *, questions=None, jev_model='typesafe-ai/jev'):
    task = validate_task(task)
    questions = validate_questions(copy.deepcopy(questions or DEFAULT_TRIAGE_QUESTIONS), strict=True)
    if set(questions['route']['criteria']) != set(ROUTES):
        raise ValueError('triage route criteria must be exactly settled, open_ended, hybrid')
    evidence = probe(task)
    state = {'objective': task['objective'], 'evidence': evidence,
             'examples': [{'observation': _clip(example['observation']), 'expected': _clip(example['expected'])}
                          for example in task['examples'][:EXAMPLE_LIMIT]]}
    response = copy.deepcopy(jev.judge(state, copy.deepcopy(questions), model=jev_model))
    answer = response['answers']['route']
    probabilities = {route: float(answer.get('probabilities', {}).get(route, 0.0)) for route in ROUTES}
    if answer.get('choice') not in ROUTES:
        raise ValueError(f"triage returned an unknown route {answer.get('choice')!r}")
    route = answer['choice']
    return {'route': route, 'probabilities': probabilities, 'confidence': probabilities[route],
            'evidence': evidence, 'questions': questions, 'response': response, 'state': state}


def route_of_spec(spec):
    kinds = {node.get('kind') for node in spec.get('nodes', [])}
    if 'jev' in kinds and 'llm' in kinds:
        return 'hybrid'
    if 'jev' in kinds:
        return 'jev'
    if 'llm' in kinds:
        return 'llm'
    return 'code'


_SETTLED_CATALOG = '''def run(obs, nodes, memory):
    options = {}
    if isinstance(obs.get("labels"), list) and obs["labels"]:
        for label in obs["labels"]:
            options[str(label)] = "The correct answer is " + str(label) + "."
    else:
        for item in obs.get(CATALOG_FIELD) or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                options[item["id"]] = str(item)
            elif isinstance(item, str):
                options[item] = item
    if not options:
        options["none"] = "No option was available."
    return {"options": list(options), "questions": {"action": {
        "type": "choice",
        "instructions": OBJECTIVE + " Choose exactly one option ID for the observation.",
        "criteria": options}}}
'''

_SETTLED_DECISION = '''def run(obs, nodes, memory):
    options = nodes["catalog"]["options"]
    chosen = nodes["pick"]["action"].get("choice")
    if chosen in options:
        return chosen
    return options[0]
'''

_OPEN_PROMPT = '''def run(obs, nodes, memory):
    lines = [OBJECTIVE, ""]
    for field in FIELDS:
        lines.append(field + ": " + str(obs.get(field)))
    lines.append("")
    lines.append("Write the best output for this input.")
    return {"prompt": "\\n".join(lines)}
'''

_OPEN_VALIDATE = '''def run(obs, nodes, memory):
    return nodes["draft"]
'''

_HYBRID_PROMPT = '''def run(obs, nodes, memory):
    lines = [OBJECTIVE, ""]
    for field in FIELDS:
        lines.append(field + ": " + str(obs.get(field)))
    lines.append("")
    lines.append("Write 3 different candidate outputs for this input.")
    return {"prompt": "\\n".join(lines)}
'''

_HYBRID_CRITERIA = '''def run(obs, nodes, memory):
    candidates = nodes["propose"]["candidates"]
    criteria = {}
    for index, candidate in enumerate(candidates):
        text = str(candidate).strip()
        criteria["c" + str(index)] = text if text else "(empty candidate)"
    return {"candidates": candidates, "questions": {"best": {
        "type": "choice",
        "instructions": OBJECTIVE + " Choose the candidate that best serves this input.",
        "criteria": criteria}}}
'''

_HYBRID_DECISION = '''def run(obs, nodes, memory):
    candidates = nodes["criteria"]["candidates"]
    chosen = nodes["select"]["best"].get("choice") or "c0"
    index = int(chosen[1:]) if chosen[1:].isdigit() and int(chosen[1:]) < len(candidates) else 0
    return {"text": str(candidates[index])}
'''


def _fill(source, task):
    fields = task['observation_fields']
    return (source.replace('OBJECTIVE', json.dumps(task['objective']))
                  .replace('FIELDS', json.dumps(fields))
                  .replace('CATALOG_FIELD', json.dumps(task['action_space'].get('catalog_field') or 'legal_actions')))


def seed_for_route(route, task):
    task = validate_task(task)
    fields_expr = '{' + ', '.join(f"{json.dumps(f)}: obs[{json.dumps(f)}]" for f in task['observation_fields']) + '}'
    if route == 'settled':
        spec = {'version': 4, 'name': f"{task['name']} · settled seed", 'jev_model': 'typesafe-ai/jev',
                'nodes': [
                    {'id': 'catalog', 'kind': 'python', 'source': _fill(_SETTLED_CATALOG, task), 'depends_on': []},
                    {'id': 'pick', 'kind': 'jev', 'state': fields_expr, 'questions_expression': "nodes['catalog']['questions']"},
                    {'id': 'decision', 'kind': 'python', 'source': _SETTLED_DECISION, 'depends_on': ['catalog', 'pick']}],
                'output': "nodes['decision']"}
    elif route == 'open_ended':
        draft = {'id': 'draft', 'kind': 'llm', 'prompt_expression': "nodes['prompt']['prompt']"}
        if 'output_schema' in task:
            draft['output_schema'] = task['output_schema']
        spec = {'version': 4, 'name': f"{task['name']} · open-ended seed", 'jev_model': 'typesafe-ai/jev', 'llm_model': 'mock-llm',
                'nodes': [
                    {'id': 'prompt', 'kind': 'python', 'source': _fill(_OPEN_PROMPT, task), 'depends_on': []},
                    draft,
                    {'id': 'validate', 'kind': 'python', 'source': _OPEN_VALIDATE, 'depends_on': ['draft']}],
                'output': "nodes['validate']"}
    elif route == 'hybrid':
        spec = {'version': 4, 'name': f"{task['name']} · hybrid seed", 'jev_model': 'typesafe-ai/jev', 'llm_model': 'mock-llm',
                'nodes': [
                    {'id': 'prompt', 'kind': 'python', 'source': _fill(_HYBRID_PROMPT, task), 'depends_on': []},
                    {'id': 'propose', 'kind': 'llm', 'prompt_expression': "nodes['prompt']['prompt']",
                     'output_schema': {'type': 'object', 'required': ['candidates'], 'properties': {
                         'candidates': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'minItems': 2, 'maxItems': 5}}}},
                    {'id': 'criteria', 'kind': 'python', 'source': _fill(_HYBRID_CRITERIA, task), 'depends_on': ['propose']},
                    {'id': 'select', 'kind': 'jev', 'state': fields_expr, 'questions_expression': "nodes['criteria']['questions']"},
                    {'id': 'decision', 'kind': 'python', 'source': _HYBRID_DECISION, 'depends_on': ['criteria', 'select']}],
                'output': "nodes['decision']"}
    else:
        raise ValueError('unknown route')
    return validate_spec(spec)
