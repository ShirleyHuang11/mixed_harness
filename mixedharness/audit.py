"""The ruler: how many of an agent's turns are closed choices a judge could take, and what that would save.

    mixedharness audit runs/tau2/<run>/llm_only/results.json --format tau2 --judge jev
    mixedharness audit traces.jsonl --format messages --judge logprob:Qwen/Qwen3-4B@http://127.0.0.1:8790/v1

Input is any recorded agent trace: a tau2 `results.json` (`--format tau2`), or JSON lines of OpenAI-style
conversations (`--format messages`: `{"id", "tools": [tool schemas], "messages": [{"role", "content",
"tool_calls"}]}`). Nothing else is needed: no domain compiler, no harness, no benchmark. For every assistant turn
the judge sees the conversation so far and the tool list, and answers two questions: is the next step a closed
choice among the listed tools and a routine reply (route), and which one (action). The agent's recorded action is
the answer key.

The report is a curve over the threshold tau: the share of turns the judge would take (route closed and action
probability >= tau), how often its pick matches what the agent did on those turns, and the cost of the trace if
the LLM were called only on the remaining turns (LLM cost per turn from the trace's own usage, judge cost from the
judge's replies). The share is what the harness can move off the LLM; the agreement is the accuracy it would pay;
the cost ratio is the saving. One caveat belongs in the reading: naming the right tool is not yet filling its
arguments, which the harness's compiler does from state, so the audit is an upper bound on what a harness built on
this judge can take, not a measurement of a built harness.
"""
import concurrent.futures
import datetime as dt
import json
import time
from pathlib import Path

from .jev import JevError
from .judge import make_judge
from .llm import PRICES

TAUS = (0.6, 0.8, 0.9, 0.95, 0.99)
HISTORY = 12
TEXT_LIMIT = 600
REPLY = 'reply'

ROUTE = {'closed': 'The next step is one of the listed tools, with every argument it needs already said in the conversation or returned by an earlier tool, or a routine reply: asking for a listed piece of information, reporting a tool result, confirming an action, or closing. A judge could choose it from the list.',
         'escalate': 'The next step needs reasoning, explanation, comparison or wording that no listed tool or routine reply covers; only a full language-model turn settles it.'}
ROUTE_INSTRUCTIONS = 'You are judging one turn of a tool-using agent. Given the conversation so far and the tools it has, decide whether its next step is a closed choice or needs a full language-model turn.'
ACTION_INSTRUCTIONS = 'Which single action should the agent take next? Choose one listed tool when its arguments are already known from the conversation; choose "reply" when the agent should write to the user instead.'


# -- trace adapters ----------------------------------------------------------------------------------------------
def _cost_of(message, price):
    cost = message.get('cost')
    if cost is not None:   # a recorded 0.0 is a price (a canned greeting costs nothing); only a missing cost falls back to usage
        return float(cost), 'trace'
    usage = message.get('usage') or {}
    pin, pout = int(usage.get('prompt_tokens') or 0), int(usage.get('completion_tokens') or 0)
    if price and (pin or pout):
        return (pin * price[0] + pout * price[1]) / 1_000_000, 'usage_x_list_price'
    return None, None


def _actual(message):
    calls = message.get('tool_calls') or []
    if calls:
        call = calls[0]
        fn = call.get('function') or call
        args = fn.get('arguments')
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {'_raw': args}
        return {'kind': 'tool', 'name': fn.get('name'), 'arguments': args or {}}
    content = message.get('content')
    if isinstance(content, list):
        content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
    return {'kind': 'message', 'text': content or ''}


def _history_item(message, names_by_call_id):
    role = message.get('role')
    if role == 'tool':
        name = names_by_call_id.get(message.get('id') or message.get('tool_call_id'))
        return {'role': 'tool', 'tool': name, 'result': str(message.get('content') or '')[:TEXT_LIMIT], 'error': bool(message.get('error'))}
    actual = _actual(message)
    if actual['kind'] == 'tool':
        return {'role': role, 'tool': actual['name'], 'arguments': actual['arguments']}
    return {'role': role, 'text': actual['text'][:TEXT_LIMIT]}


def conversations_from_tau2(path, price=None):
    data = json.loads(Path(path).read_text())
    sims = data.get('simulations') if isinstance(data, dict) else data
    out = []
    for sim in sims or []:
        turns, history, names = [], [], {}
        for m in sim.get('messages') or []:
            if m.get('role') == 'assistant':
                cost, priced_by = _cost_of(m, price)
                turns.append({'history': list(history[-HISTORY:]), 'actual': _actual(m), 'llm_cost_usd': cost, 'priced_by': priced_by,
                              'prompt_tokens': int((m.get('usage') or {}).get('prompt_tokens') or 0)})
                for call in m.get('tool_calls') or []:
                    names[call.get('id')] = call.get('name')
            history.append(_history_item(m, names))
        reward = (sim.get('reward_info') or {}).get('reward')
        out.append({'id': str(sim.get('task_id', len(out))), 'turns': turns, 'reward': reward})
    return out


def conversations_from_messages(path, price=None):
    text = Path(path).read_text()
    records = [json.loads(line) for line in text.splitlines() if line.strip()] if path.endswith('.jsonl') else json.loads(text)
    if isinstance(records, dict):
        records = [records]
    out = []
    for i, rec in enumerate(records):
        messages = rec.get('messages') if isinstance(rec, dict) else rec
        tools = {}
        for t in (rec.get('tools') if isinstance(rec, dict) else None) or []:
            fn = t.get('function') or t
            tools[fn.get('name')] = str(fn.get('description') or '')[:300]
        turns, history, names = [], [], {}
        for m in messages or []:
            if m.get('role') == 'assistant':
                cost, priced_by = _cost_of(m, price)
                turns.append({'history': list(history[-HISTORY:]), 'actual': _actual(m), 'llm_cost_usd': cost, 'priced_by': priced_by,
                              'prompt_tokens': int((m.get('usage') or {}).get('prompt_tokens') or 0)})
                for call in m.get('tool_calls') or []:
                    fn = call.get('function') or call
                    names[call.get('id')] = fn.get('name')
            elif m.get('role') == 'system':
                continue
            history.append(_history_item(m, names))
        out.append({'id': str(rec.get('id', i)) if isinstance(rec, dict) else str(i), 'turns': turns, 'tools': tools, 'reward': rec.get('reward') if isinstance(rec, dict) else None})
    return out


def tools_seen(conversations):
    names = set()
    for c in conversations:
        for t in c['turns']:
            if t['actual']['kind'] == 'tool' and t['actual'].get('name'):
                names.add(t['actual']['name'])
        names |= set((c.get('tools') or {}).keys())
    return sorted(names)


# -- the judgement --------------------------------------------------------------------------------------------------
def options_for(tools):
    options = {'tool:' + name: (desc or 'call this tool') for name, desc in tools.items()}
    options[REPLY] = 'Write a message to the user: ask for information, report a result, confirm an action, or close the conversation.'
    return options


def judge_turn(judge, turn, tools, policy_text=None):
    options = options_for(tools)
    state = {'tools': tools, 'conversation': turn['history'], 'options': options}
    if policy_text:
        state['policy'] = policy_text
    questions = {'route': {'type': 'choice', 'instructions': ROUTE_INSTRUCTIONS, 'criteria': ROUTE},
                 'action': {'type': 'choice', 'instructions': ACTION_INSTRUCTIONS, 'criteria': options}}
    response = judge.judge(state, questions)
    route, action = response['answers']['route'], response['answers']['action']
    actual = turn['actual']
    actual_id = ('tool:' + str(actual.get('name'))) if actual['kind'] == 'tool' else REPLY
    pick = action['choice']
    return {'route': route['choice'], 'route_p': route['probabilities'].get(route['choice']), 'pick': pick, 'pick_p': action['probabilities'].get(pick),
            'actual': actual_id, 'agree': pick == actual_id, 'judge_cost_usd': response.get('cost_usd') or 0.0, 'judge_ms': response.get('elapsed_ms')}


def curve(rows, taus=TAUS):
    """Per tau: share of turns taken, agreement on them, and the cost ratio of (LLM on the rest + judge on all) to LLM on all."""
    llm_total = sum(r['llm_cost_usd'] or 0.0 for r in rows)
    priced = all(r.get('llm_cost_usd') is not None for r in rows) and llm_total > 0
    judge_total = sum(r['judge_cost_usd'] for r in rows)
    out = []
    for tau in taus:
        taken = [r for r in rows if r['route'] == 'closed' and (r['pick_p'] or 0.0) >= tau]
        agree = sum(1 for r in taken if r['agree'])
        kept = sum(r['llm_cost_usd'] or 0.0 for r in rows if r not in taken)
        out.append({'tau': tau, 'turns_taken': len(taken), 'share_taken': len(taken) / len(rows) if rows else 0.0,
                    'agreement': agree / len(taken) if taken else None,
                    'tool_turns_taken': sum(1 for r in taken if r['actual'] != REPLY), 'reply_turns_taken': sum(1 for r in taken if r['actual'] == REPLY),
                    'cost_ratio': ((kept + judge_total) / llm_total) if priced else None})
    return {'rows': out, 'llm_cost_total_usd': llm_total if priced else None, 'judge_cost_total_usd': judge_total, 'priced': priced}


def run_audit(paths, *, fmt='tau2', judge_spec='jev', tools=None, price_model=None, out_dir=None, workers=4, limit=None, cache_dir=None, policy_text=None, namespace=None, judge_temperature=1.0):
    price = next((v for k, v in PRICES.items() if price_model and price_model.startswith(k)), None)
    loader = conversations_from_tau2 if fmt == 'tau2' else conversations_from_messages
    conversations = [c for p in paths for c in loader(p, price)]
    if limit:
        conversations = conversations[:limit]
    tool_map = dict(tools or {})
    for name in tools_seen(conversations):
        tool_map.setdefault(name, '')
    for c in conversations:
        for name, desc in (c.get('tools') or {}).items():
            if desc and not tool_map.get(name):
                tool_map[name] = desc
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    judge = make_judge(judge_spec, cache_dir=cache_dir, cache_namespace=namespace or f'audit-{stamp}', temperature=judge_temperature)
    jobs = [(c, i, t) for c in conversations for i, t in enumerate(c['turns'])]
    rows = [None] * len(jobs)
    started = time.perf_counter()

    def work(k):
        c, i, t = jobs[k]
        try:
            verdict = judge_turn(judge, t, tool_map, policy_text)
        except JevError as exc:
            verdict = {'route': None, 'route_p': None, 'pick': None, 'pick_p': None, 'actual': None, 'agree': None, 'judge_cost_usd': 0.0, 'judge_ms': None, 'error': str(exc)[:200]}
        rows[k] = {'conversation': c['id'], 'turn': i + 1, 'reward': c.get('reward'), 'llm_cost_usd': t['llm_cost_usd'], 'priced_by': t['priced_by'],
                   'prompt_tokens': t['prompt_tokens'], **verdict}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(len(jobs))))
    good = [r for r in rows if r.get('route') is not None]
    report = {'judge': judge_spec, 'judge_temperature': judge_temperature, 'format': fmt, 'paths': [str(p) for p in paths], 'conversations': len(conversations), 'turns': len(rows), 'judged': len(good),
              'errors': len(rows) - len(good), 'tools': sorted(tool_map), 'price_model': price_model,
              'route_closed_share': sum(1 for r in good if r['route'] == 'closed') / len(good) if good else None,
              'agreement_all_turns': sum(1 for r in good if r['agree']) / len(good) if good else None,
              'agreement_route_closed': (lambda s: sum(1 for r in s if r['agree']) / len(s) if s else None)([r for r in good if r['route'] == 'closed']),
              'curve': curve(good), 'judge_stats': judge.stats, 'judge_metadata': judge.metadata, 'elapsed_s': round(time.perf_counter() - started, 1),
              'caveat': 'Upper bound: the judge names the tool; a harness still has to fill its arguments from state (the compiler does) and keep a guard.'}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'audit.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + '\n')
        (out / 'report.json').write_text(json.dumps(report, indent=1, default=str))
        (out / 'report.md').write_text(render(report))
    return report, rows


def render(report):
    c = report['curve']
    lines = [f"# Turn audit: {report['conversations']} conversations, {report['turns']} assistant turns, judge `{report['judge']}`" + (f" at temperature {report['judge_temperature']}" if report.get('judge_temperature', 1.0) != 1.0 else ''), '',
             f"The judge's pick matches the agent's recorded action on {report['agreement_all_turns']:.0%} of all turns. "
             f"Its route answer says closed on {report['route_closed_share']:.0%} of turns"
             + (" (with every tool listed and no compiler to say which arguments are known, the route question carries little signal here; the action probability does the work)." if (report['route_closed_share'] or 0) > 0.95 else '.'), '',
             '| tau | turns the judge takes | of which tool / reply | pick = agent action | cost of the trace |', '|---|---|---|---|---|']
    for r in c['rows']:
        cost = f"{r['cost_ratio']:.2f}x" if r['cost_ratio'] is not None else 'unpriced'
        agree = f"{r['agreement']:.0%}" if r['agreement'] is not None else '—'
        lines.append(f"| {r['tau']:.2f} | {r['turns_taken']} ({r['share_taken']:.0%}) | {r['tool_turns_taken']} / {r['reply_turns_taken']} | {agree} | {cost} |")
    if c['priced']:
        lines += ['', f"LLM cost of the trace as recorded: ${c['llm_cost_total_usd']:.2f}; the judge's cost over every turn: ${c['judge_cost_total_usd']:.4f}."]
    else:
        lines += ['', 'The trace carries no per-turn cost and no --price-model matched, so the cost column is empty; shares and agreement still hold.']
    if report['errors']:
        lines += ['', f"{report['errors']} turns failed at the judge and are left out."]
    lines += ['', 'Reading: the share is what a harness built on this judge could move off the LLM at that threshold; the agreement is the accuracy it would pay on those turns; the cost column is the trace re-priced with the LLM called only on the remaining turns.',
              report['caveat'], '']
    return '\n'.join(lines)
