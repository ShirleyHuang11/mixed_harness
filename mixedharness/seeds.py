"""Generic seed harnesses for a task described as JSON, and the shape test `route_of_spec`.

TaskDescription (JSON):

    {"name": str, "objective": str,
     "observation_fields": [str],
     "action_space": {"kind": "labels"|"catalog"|"open"|"declared_unknown", "labels": [str]?, "catalog_field": str?},
     "evaluator": {"kind": "exact_match"|"environment"|"code_check"|"reward_model"|"judge"},
     "decisions_per_episode": int, "latency_budget_ms": number|null,
     "output_schema": {...}?,
     "examples": [{"observation": {...}, "expected": any}, ...]}

Two generic seeds, one per path family, for single-step tasks (the bundled examples):

    jev   catalog (python) -> pick (Jev) -> decision (python): code enumerates the options, Jev chooses
    llm   prompt (python) -> draft (llm) for single-shot outputs; context (python) -> agent for tool tasks

Multi-step tool tasks need a domain compiler; `examples/tau2/seeds.py` is the worked one, with the three paths and
the guard. `route_of_spec` reports what a finished spec is made of: jev, llm, jev+llm or code.
"""
import copy
import json

from .expressions import check_json
from .spec import FORMAT, validate_spec

ROUTES = ('jev', 'llm')
ACTION_KINDS = ('labels', 'catalog', 'open', 'declared_unknown')
EVALUATOR_KINDS = ('exact_match', 'environment', 'code_check', 'reward_model', 'judge')


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
    return {"options": list(options), "questions": {"action": {
        "type": "choice",
        "instructions": __OBJECTIVE__ + " Choose exactly one option id for the observation.",
        "criteria": options}}}
'''

_DECISION = '''def run(obs, nodes, memory):
    options = nodes["catalog"]["options"]
    chosen = nodes["pick"]["action"].get("choice")
    if chosen in options:
        return {"action": chosen}
    return {"action": options[0]}
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

AGENT_INSTRUCTIONS = ('You are the agent for this task. Follow the policy. In each turn either call exactly one tool or send one '
                      'message; never invent facts the tools did not return.')


def _fill(source, task):
    labels = task['action_space'].get('labels') if task['action_space']['kind'] == 'labels' else []
    return (source.replace('__OBJECTIVE__', json.dumps(task['objective']))
                  .replace('__FIELDS__', json.dumps(task['observation_fields']))
                  .replace('__LABELS__', json.dumps(list(labels or [])))
                  .replace('__CATALOG_FIELD__', json.dumps(task['action_space'].get('catalog_field') or 'legal_actions')))


def seed_for_route(route, task, *, jev_model='jev-1.13.0', llm_model='mock-llm'):
    task = validate_task(task)
    fields_expr = '{' + ', '.join(f"{json.dumps(f)}: get(obs, {json.dumps(f)})" for f in task['observation_fields']) + '}'
    tool_task = task['action_space']['kind'] == 'catalog'
    if route == 'jev':
        nodes = [{'id': 'catalog', 'kind': 'python', 'source': _fill(_CATALOG, task), 'inputs': []},
                 {'id': 'pick', 'kind': 'jev', 'state': fields_expr, 'questions_expression': "nodes['catalog']['questions']"},
                 {'id': 'decision', 'kind': 'python', 'source': _DECISION, 'inputs': ['catalog', 'pick']}]
        spec = {'format': FORMAT, 'name': f"{task['name']} · jev", 'jev_model': jev_model, 'nodes': nodes, 'output': "nodes['decision']['action']"}
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
        spec = {'format': FORMAT, 'name': f"{task['name']} · llm", 'jev_model': jev_model, 'llm_model': llm_model, 'nodes': nodes, 'output': output}
    else:
        raise TaskError('route must be one of ' + ', '.join(ROUTES))
    return validate_spec(spec)
