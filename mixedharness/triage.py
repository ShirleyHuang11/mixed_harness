"""Triage: which harness shape fits a task. Code probes the description, Jev decides, one seed is built.

TaskDescription (JSON):

    {"name": str, "objective": str,
     "observation_fields": [str],
     "action_space": {"kind": "labels"|"catalog"|"open"|"declared_unknown", "labels": [str]?, "catalog_field": str?},
     "evaluator": {"kind": "exact_match"|"environment"|"code_check"|"reward_model"|"judge"},
     "decisions_per_episode": int, "latency_budget_ms": number|null,
     "output_schema": {...}?,
     "examples": [{"observation": {...}, "expected": any}, ...]}

The triage is itself a harness, one level up, and runs through the same Runtime as any task harness:
`triage_spec()` is a spec with a python probe node (nine deterministic fields from the task description), one
Jev node (the route question plus two auxiliary questions), and a python decision node (the argmax route).
`triage(task, jev)` runs it and returns a RouteDecision: the route, the full distribution, the evidence, the
exact Jev state and response, and the hash of the triage spec that produced it. `probe(task)` is the
in-process reference implementation of the probe node, kept identical for tests. `seed_for_route(route, task)`
builds the seed spec for that route.

Routes:  jev (code computes the facts, Jev makes the bounded choices, code acts);
         llm (one language-model call per decision: an agent turn with tools, or a draft for single-shot tasks).
Evolution may add a hand_to_llm option and an agent node to a jev route; `route_of_spec` reports that as 'jev+llm'.
"""
import copy
import json
import statistics

from .expressions import check_json
from .spec import FORMAT, spec_hash, validate_questions, validate_spec

ROUTES = ('jev', 'llm')
ACTION_KINDS = ('labels', 'catalog', 'open', 'declared_unknown')
EVALUATOR_KINDS = ('exact_match', 'environment', 'code_check', 'reward_model', 'judge')
EXACT_EVALUATORS = ('exact_match', 'environment', 'code_check')
SHORT_TEXT_TOKENS = 12
EXAMPLE_LIMIT = 5
EXAMPLE_CHARS = 300
# What a finished harness looks like, by node kinds present → the triage vocabulary it corresponds to.
TARGET_OF = {'jev': 'jev', 'llm': 'llm', 'jev+llm': 'jev', 'code': 'jev'}

DEFAULT_TRIAGE_QUESTIONS = {
    'route': {
        'type': 'choice',
        'instructions': ('Using the probe evidence and examples, judge which route fits this task. '
                         'Evidence fields are computed by code from the task description; examples are representative.'),
        'criteria': {
            'jev': 'Most decisions pick from a finite, enumerable set (labels or a per-step legal action list) and code can '
                   'compute the facts. Code plus a judgment model; a language model at most on rare decisions.',
            'llm': 'Most decisions need free text, multi-step reasoning or composition that only a language model can produce; '
                   'a full model call per decision is the right tool.',
        },
    },
    'code_computes_facts': {'type': 'noul', 'instructions': 'Can deterministic code compute most facts needed for each decision, leaving only a bounded judgment to a model?'},
    'needs_multistep': {'type': 'noul', 'instructions': 'Does a single decision require multi-step reasoning or tool use before an answer can be given?'},
}


class TaskError(ValueError):
    pass


def validate_task(task):
    if not isinstance(task, dict):
        raise TaskError('task must be an object')
    check_json(task)
    for key in ('name', 'objective', 'observation_fields', 'action_space', 'evaluator', 'decisions_per_episode', 'examples'):
        if key not in task:
            raise TaskError(f'task requires {key}')
    if not isinstance(task['observation_fields'], list) or not all(isinstance(f, str) for f in task['observation_fields']):
        raise TaskError('observation_fields must be a list of strings')
    if not isinstance(task['examples'], list) or not task['examples']:
        raise TaskError('examples must be a nonempty list')
    for example in task['examples']:
        if not isinstance(example, dict) or 'observation' not in example or 'expected' not in example:
            raise TaskError('each example needs observation and expected')
    action = task['action_space']
    if not isinstance(action, dict) or action.get('kind') not in ACTION_KINDS:
        raise TaskError('action_space.kind must be one of ' + ', '.join(ACTION_KINDS))
    if action['kind'] == 'labels' and (not isinstance(action.get('labels'), list) or not action['labels']):
        raise TaskError('a labels action space needs a nonempty labels list')
    if action['kind'] == 'catalog' and not isinstance(action.get('catalog_field'), str):
        raise TaskError('a catalog action space needs catalog_field')
    if not isinstance(task['evaluator'], dict) or task['evaluator'].get('kind') not in EVALUATOR_KINDS:
        raise TaskError('evaluator.kind must be one of ' + ', '.join(EVALUATOR_KINDS))
    n = task['decisions_per_episode']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise TaskError('decisions_per_episode must be a positive integer')
    if 'output_schema' in task and not isinstance(task['output_schema'], dict):
        raise TaskError('output_schema must be an object')
    return copy.deepcopy(task)


def _tokens(value):
    return len(str(value).split()) if isinstance(value, str) else len(str(value))


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
        expected_kind = 'short_text' if statistics.median(_tokens(v) for v in expected) <= SHORT_TEXT_TOKENS else 'long_text'
    else:
        expected_kind = 'structured'
    return {
        'declared_action_kind': action['kind'],
        'label_count': len(labels) if labels is not None else None,
        'catalog_present_in_examples': bool(field) and all(
            isinstance(ex['observation'].get(field), list) and ex['observation'][field] for ex in task['examples']),
        'exact_validator': task['evaluator']['kind'] in EXACT_EVALUATORS,
        'expected_kind': expected_kind,
        'median_expected_tokens': statistics.median(_tokens(v) for v in expected),
        'decisions_per_episode': task['decisions_per_episode'],
        'latency_budget_ms': task.get('latency_budget_ms'),
        'example_count': len(task['examples']),
    }


def _clip(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= EXAMPLE_CHARS else text[:EXAMPLE_CHARS] + '…'


def validate_triage_questions(questions):
    questions = validate_questions(copy.deepcopy(questions))
    if 'route' not in questions or questions['route']['type'] != 'choice' or set(questions['route']['criteria']) != set(ROUTES):
        raise TaskError('triage questions need a route choice whose criteria are exactly jev, llm')
    return questions


# ---------------------------------------------------------------------------------------------------------
# The triage is itself a harness, one level up: a python probe, one Jev question set, a python decision.
# It runs through the same Runtime as any task harness, on the task description instead of an observation.
# ---------------------------------------------------------------------------------------------------------
PROBE_SOURCE = '''def tokens(value):
    if isinstance(value, str):
        return len(value.split())
    return len(str(value))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def run(obs, nodes, memory):
    task = obs["task"]
    action = task["action_space"]
    labels = action.get("labels") if action["kind"] == "labels" else None
    field = action.get("catalog_field")
    examples = task["examples"]
    expected = [ex["expected"] for ex in examples]
    if labels is not None and all(v in labels for v in expected):
        kind = "label"
    elif all(isinstance(v, str) for v in expected):
        kind = "short_text" if median([tokens(v) for v in expected]) <= __SHORT_TEXT_TOKENS__ else "long_text"
    else:
        kind = "structured"
    catalog = bool(field) and all(isinstance(ex["observation"].get(field), list) and len(ex["observation"].get(field)) > 0 for ex in examples)
    return {"declared_action_kind": action["kind"],
            "label_count": len(labels) if labels is not None else None,
            "catalog_present_in_examples": catalog,
            "exact_validator": task["evaluator"]["kind"] in __EXACT_EVALUATORS__,
            "expected_kind": kind,
            "median_expected_tokens": median([tokens(v) for v in expected]),
            "decisions_per_episode": task["decisions_per_episode"],
            "latency_budget_ms": task.get("latency_budget_ms"),
            "example_count": len(examples)}
'''

DECISION_SOURCE = '''def run(obs, nodes, memory):
    answer = nodes["route"]["route"]
    routes = __ROUTES__
    probabilities = {}
    for r in routes:
        probabilities[r] = float((answer.get("probabilities") or {}).get(r, 0.0))
    route = answer.get("choice")
    if route not in routes:
        best = routes[0]
        for r in routes:
            if probabilities[r] > probabilities[best]:
                best = r
        route = best
    return {"route": route, "probabilities": probabilities, "confidence": probabilities[route]}
'''


def triage_spec(questions=None, *, jev_model='jev-1.13.0'):
    """The triage as a spec: probe (python) -> route (Jev) -> decision (python)."""
    questions = validate_triage_questions(questions or DEFAULT_TRIAGE_QUESTIONS)
    probe_source = (PROBE_SOURCE.replace('__SHORT_TEXT_TOKENS__', str(SHORT_TEXT_TOKENS))
                    .replace('__EXACT_EVALUATORS__', json.dumps(list(EXACT_EVALUATORS))))
    decision_source = DECISION_SOURCE.replace('__ROUTES__', json.dumps(list(ROUTES)))
    return validate_spec({'format': FORMAT, 'name': 'triage', 'jev_model': jev_model, 'nodes': [
        {'id': 'probe', 'kind': 'python', 'source': probe_source, 'inputs': []},
        {'id': 'route', 'kind': 'jev', 'state': "{'objective': obs['task']['objective'], 'evidence': nodes['probe'], 'examples': obs['examples']}",
         'questions': questions},
        {'id': 'decision', 'kind': 'python', 'source': decision_source, 'inputs': ['route']}],
        'output': "nodes['decision']"})


def triage_observation(task):
    task = validate_task(task)
    return {'task': {k: task[k] for k in ('name', 'objective', 'observation_fields', 'action_space', 'evaluator', 'decisions_per_episode', 'examples')}
                     | ({'latency_budget_ms': task['latency_budget_ms']} if 'latency_budget_ms' in task else {}),
            'examples': [{'observation': _clip(ex['observation']), 'expected': _clip(ex['expected'])} for ex in task['examples'][:EXAMPLE_LIMIT]]}


def triage(task, jev, *, questions=None, jev_model=None):
    """Run the triage harness on a task description. Returns the RouteDecision."""
    from .runtime import Runtime
    spec = triage_spec(questions, jev_model=jev_model or getattr(jev, 'model', None) or 'jev-1.13.0')
    runtime = Runtime(spec, jev)
    try:
        result = runtime.run(triage_observation(task))
    finally:
        runtime.close()
    records = {r['id']: r for r in result['trace']}
    decision = result['output']
    return {'route': decision['route'], 'probabilities': decision['probabilities'], 'confidence': decision['confidence'],
            'evidence': result['nodes']['probe'], 'questions': spec['nodes'][1]['questions'],
            'response': records['route']['response'], 'state': records['route']['state'], 'spec_hash': spec_hash(spec)}


def route_of_spec(spec):
    """What a spec is made of: jev / llm / jev+llm / code. An agent or llm node is a language-model call."""
    kinds = {node.get('kind') for node in spec.get('nodes', [])}
    model = bool(kinds & {'llm', 'agent'})
    if 'jev' in kinds and model:
        return 'jev+llm'
    if 'jev' in kinds:
        return 'jev'
    if model:
        return 'llm'
    return 'code'


# ---------------------------------------------------------------------------------------------------------
# Seed templates. Placeholders are replaced with JSON literals from the task description.
#   jev route:  catalog (python) -> pick (Jev) -> decision (python)
#   llm route:  prompt (python) -> draft (llm) for single-shot tasks; context (python) -> agent for tool tasks
#   Evolution may later add a hand_to_llm catalog option and an agent node to a jev route; that is not a third route.
# ---------------------------------------------------------------------------------------------------------
_CATALOG = '''def run(obs, nodes, memory):
    options = {}
    labels = obs.get("labels") if isinstance(obs.get("labels"), list) and obs.get("labels") else __LABELS__
    if labels:
        for label in labels:
            options[str(label)] = "The correct answer is " + str(label) + "."
    else:
        for item in obs.get(__CATALOG_FIELD__) or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                options[item["id"]] = str(item)
            elif isinstance(item, str):
                options[item] = item
    if not options:
        options["none"] = "No option was available."
    if __HAND_TO_LLM__:
        options["hand_to_llm"] = "None of the options above is right, or the answer has to be written rather than selected; hand this decision to the language model."
    return {"options": list(options), "questions": {"action": {
        "type": "choice",
        "instructions": __OBJECTIVE__ + " Choose exactly one option id for the observation.",
        "criteria": options}}}
'''

_DECISION = '''def run(obs, nodes, memory):
    options = nodes["catalog"]["options"]
    chosen = nodes["pick"]["action"].get("choice")
    if chosen == "hand_to_llm":
        return {"action": None, "handed": True}
    if chosen in options:
        return {"action": chosen, "handed": False}
    return {"action": options[0], "handed": False}
'''

_PROMPT = '''def run(obs, nodes, memory):
    lines = [__OBJECTIVE__, ""]
    for field in __FIELDS__:
        lines.append(field + ": " + str(obs.get(field)))
    lines.append("")
    lines.append("Write the best output for this input.")
    return {"prompt": "\\n".join(lines)}
'''

_CONTEXT = '''def run(obs, nodes, memory):
    tools = []
    for name, spec in (obs.get("tools") or {}).items():
        properties = {}
        for pname, p in (spec.get("params") or {}).items():
            properties[pname] = {"type": p.get("type", "string"), "description": p.get("description", "")}
        tools.append({"name": name, "description": spec.get("description", ""),
                      "parameters": {"type": "object", "properties": properties, "required": list(spec.get("required") or [])}})
    conversation = list(obs.get("conversation") or [])
    if obs.get("incoming"):
        conversation.append(obs["incoming"])
    return {"policy": obs.get("policy") or __OBJECTIVE__, "tools": tools, "conversation": conversation}
'''


def _fill(source, task, *, hand_to_llm=False):
    labels = task['action_space'].get('labels') if task['action_space']['kind'] == 'labels' else []
    return (source.replace('__OBJECTIVE__', json.dumps(task['objective']))
                  .replace('__FIELDS__', json.dumps(task['observation_fields']))
                  .replace('__LABELS__', json.dumps(list(labels or [])))
                  .replace('__CATALOG_FIELD__', json.dumps(task['action_space'].get('catalog_field') or 'legal_actions'))
                  .replace('__HAND_TO_LLM__', 'True' if hand_to_llm else 'False'))


AGENT_INSTRUCTIONS = ('You are the agent for this task. Follow the policy. In each turn either call exactly one tool or send one '
                      'message; never invent facts the tools did not return.')


def _jev_nodes(task, fields_expr, *, hand_to_llm):
    return [
        {'id': 'catalog', 'kind': 'python', 'source': _fill(_CATALOG, task, hand_to_llm=hand_to_llm), 'inputs': []},
        {'id': 'pick', 'kind': 'jev', 'state': fields_expr, 'questions_expression': "nodes['catalog']['questions']"},
        {'id': 'decision', 'kind': 'python', 'source': _DECISION, 'inputs': ['catalog', 'pick']}]


def seed_for_route(route, task, *, jev_model='jev-1.13.0', llm_model='mock-llm'):
    task = validate_task(task)
    fields_expr = '{' + ', '.join(f"{json.dumps(f)}: get(obs, {json.dumps(f)})" for f in task['observation_fields']) + '}'
    tool_task = task['action_space']['kind'] == 'catalog'
    if route == 'jev':
        spec = {'format': FORMAT, 'name': f"{task['name']} · jev route", 'jev_model': jev_model,
                'nodes': _jev_nodes(task, fields_expr, hand_to_llm=False), 'output': "nodes['decision']['action']"}
    elif route == 'llm':
        if tool_task:
            nodes = [{'id': 'context', 'kind': 'python', 'source': _fill(_CONTEXT, task), 'inputs': []},
                     {'id': 'agent', 'kind': 'agent', 'instructions': AGENT_INSTRUCTIONS, 'context': "nodes['context']"}]
            output = "nodes['agent']"
        else:
            draft = {'id': 'draft', 'kind': 'llm', 'prompt': "nodes['prompt']['prompt']"}
            if 'output_schema' in task:
                draft['schema'] = task['output_schema']
            nodes = [{'id': 'prompt', 'kind': 'python', 'source': _fill(_PROMPT, task), 'inputs': []}, draft]
            output = "nodes['draft']"
        spec = {'format': FORMAT, 'name': f"{task['name']} · llm route", 'jev_model': jev_model, 'llm_model': llm_model,
                'nodes': nodes, 'output': output}
    else:
        raise TaskError('route must be one of ' + ', '.join(ROUTES))
    return validate_spec(spec)
