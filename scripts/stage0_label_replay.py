"""Stage 0: label LLM-only turns closed or open, then replay the compiler and Jev on them.

The load-bearing number of the design is X, the fraction of steps in a real task that are closed: the legal action
set is enumerable from state and every argument of the action the LLM took can be filled from state. This script
measures X on the LLM-only trajectories already on disk, scores the compiler's closed flag against that label, and
replays the Jev request (route pick, action pick, reordered paraphrase, fits) on every turn with real Jev, so the
questions "does Jev pick what the LLM did on closed turns" and "does confidence or agreement predict its misses"
get direct measurements before any routed benchmark run.

Ground truth is what the LLM actually did, independent of our compiler (docs/design/tau2-four-arms.md, stage 0):
  tool call   closed iff its option id is in the compiled catalog and every argument value appears in the state
              (memory or conversation) at that point; otherwise open, with the reason (not_in_catalog / argument)
  text reply  closed iff it matches a template intent by rule (ask_identity, ask_order, confirm_action,
              report_result, report_error, deny, closing, transfer_notice); otherwise open. Every text label is
              written with needs_review=true for the hand check; the report counts them both ways.

Pre-registered numbers: X predicted 60-70% on retail; claim dead below 30%. Positive control: Jev's pick on
identity-lookup turns must agree with the LLM at >= 90%, else the state slice or the question is broken and
nothing else in the report is read.

Run inside the tau2-bench virtualenv:
  ~/tau2-bench/.venv/bin/python scripts/stage0_label_replay.py                       # every runs/tau2/*/llm_only/results.json
  ~/tau2-bench/.venv/bin/python scripts/stage0_label_replay.py --results runs/tau2/<stamp>/llm_only/results.json --mock

Outputs runs/stage0/<stamp>/{labels.jsonl, replay.jsonl, report.json, report.md}.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.tau2 import adapter, seeds  # noqa: E402
from mixedharness.jev import JevClient  # noqa: E402
from mixedharness.runtime import Runtime  # noqa: E402
from mixedharness.spec import FORMAT, validate_spec  # noqa: E402

TEMPLATES = ('greeting', 'ask_identity', 'ask_order', 'confirm_action', 'report_result', 'report_error', 'deny', 'closing', 'transfer_notice')
IDENTITY_TOOLS = ('find_user_id_by_name_zip', 'find_user_id_by_email')
X_KILL = 0.30
POSITIVE_CONTROL_MIN = 0.90


def compile_spec():
    """A spec with only the compiler node, so its catalog, flag, cluster and questions can be read turn by turn."""
    return validate_spec({'format': FORMAT, 'name': 'stage0 compiler', 'jev_model': seeds.JEV_MODEL,
                          'nodes': [{'id': 'compile', 'kind': 'python', 'source': seeds._fill(seeds.COMPILE_SOURCE, allow_llm=True), 'inputs': []}],
                          'output': "nodes['compile']", 'memory': "nodes['compile']['memory']"})


def template_of(text):
    t = (text or '').lower()
    if len(t) < 80 and ('how can i help' in t or 'how may i help' in t):
        return 'greeting'
    if 'transferred to a human' in t:
        return 'transfer_notice'
    if 'confirm' in t and ('yes' in t or 'proceed' in t or 'go ahead' in t):
        return 'confirm_action'
    if ('email' in t or 'zip' in t or 'full name' in t) and ('verify' in t or 'provide' in t or 'could you' in t or 'please' in t) and 'order' not in t[:80]:
        return 'ask_identity'
    if ('order id' in t or 'order number' in t or 'which order' in t) and '?' in t:
        return 'ask_order'
    if ('could not' in t or "couldn't" in t or 'unable to find' in t or 'not able to find' in t) and ('check' in t or 'again' in t):
        return 'report_error'
    if ("can't" in t or 'cannot' in t or 'unable' in t or 'not able' in t) and ('policy' in t or 'unfortunately' in t or 'sorry' in t):
        return 'deny'
    if ('great day' in t or "you're welcome" in t or 'anything else' in t or 'glad i could help' in t) and len(t) < 260:
        return 'closing'
    if '#w' in t or '$' in t or 'variant' in t or 'status' in t or 'tracking' in t or 'items' in t:
        return 'report_result'
    return None


def option_id_for(call, options):
    name = call['name']
    args = call.get('arguments') or {}
    candidates = ['tool:' + name]
    for key in ('order_id', 'product_id'):
        if args.get(key):
            candidates.append('tool:' + name + ':' + str(args[key]))
    for c in reversed(candidates):
        if c in options:
            return c, True
    return candidates[-1], False


CODE_FILLED_ARGS = {'summary'}                                    # free text the compiler composes itself
ENUM_ARGS = {'reason': ('no longer needed', 'ordered by mistake')}  # policy-fixed values Jev picks from


def args_in_state(args, memory, conversation):
    haystack = json.dumps(memory, default=str) + '\n' + json.dumps(conversation, default=str)
    missing = []
    for key, value in (args or {}).items():
        if key in CODE_FILLED_ARGS:
            continue
        if key in ENUM_ARGS:
            if str(value) not in ENUM_ARGS[key]:
                missing.append(key)
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            if v in (None, '', []):
                continue
            forms = {str(v), str(v).lstrip('#'), '#' + str(v).lstrip('#')}   # the compiler normalises order ids to a # prefix
            if not any(form in haystack for form in forms):
                missing.append(key)
                break
    return missing


def walk(sim, tool_schemas, policy, jev, runtime, max_conversation=40):
    """One LLM-only simulation -> one labelled, replayed row per assistant turn."""
    from tau2.data_model.message import AssistantMessage
    from examples.tau2.agent import parse_message
    tool_names_by_id = {}
    history, memory, rows = [], {}, []
    turn = 0
    for message in sim.messages or []:
        if isinstance(message, AssistantMessage):
            turn += 1
            incoming = history[-1] if history else {'role': 'user', 'text': '', 'tool': None, 'arguments': None, 'result': None, 'error': False}
            conversation = list(history[:-1][-max_conversation:]) if history else []
            obs = {'turn': turn, 'policy': policy, 'tools': json.loads(json.dumps(tool_schemas)), 'conversation': conversation, 'incoming': incoming}
            compiled = runtime.run(obs, memory=memory)
            cx = compiled['output']
            memory_before = compiled['memory_before']
            memory = compiled['memory']
            if message.tool_calls:
                call = message.tool_calls[0]
                llm_action = {'kind': 'tool', 'name': call.name, 'arguments': dict(call.arguments or {})}
                option_id, in_catalog = option_id_for(llm_action, cx['options'])
                missing = args_in_state(llm_action['arguments'], memory_before, conversation + [incoming])
                needs = cx['needs'].get(option_id) or []
                closed = in_catalog and not missing and not needs
                reason = None if closed else ('not_in_catalog' if not in_catalog else ('argument:' + ','.join(missing or needs)))
                target = option_id
                needs_review = not in_catalog
            else:
                template = template_of(message.content)
                llm_action = {'kind': 'message', 'text': message.content or ''}
                closed = template is not None and ('reply:' + template) in cx['options']
                reason = None if closed else ('template_not_legal:' + template if template else 'open_text')
                target = ('reply:' + template) if template else None
                needs_review = True
            if cx.get('questions') is None:      # a structural default: code decided, no Jev call
                answers, act, alt, route = {}, {'choice': cx['default'], 'probabilities': {cx['default']: 1.0}}, {}, {'choice': 'structural'}
            else:
                answers = jev.judge(cx['state'], cx['questions'], model=seeds.JEV_MODEL)['answers']
                act = answers.get('action') or {}
                alt = answers.get('action_alt') or {}
                route = answers.get('route') or {}
            pick = act.get('choice')
            rows.append({'task_id': str(sim.task_id), 'reward': float(sim.reward_info.reward) if sim.reward_info is not None and sim.reward_info.reward is not None else None,
                         'turn': turn, 'incoming_role': incoming.get('role'), 'cluster': cx['cluster'], 'compiler_closed': cx['closed'],
                         'options': list(cx['options']), 'needs': cx['needs'],
                         'llm_action': llm_action, 'label_closed': bool(closed), 'label_reason': reason, 'target': target, 'needs_review': needs_review,
                         'jev_route': route.get('choice'), 'jev_route_confidence': (route.get('probabilities') or {}).get(route.get('choice')),
                         'jev_action': pick, 'jev_confidence': (act.get('probabilities') or {}).get(pick), 'jev_alt': alt.get('choice'),
                         'agreement': (pick == alt.get('choice')) if pick is not None and alt.get('choice') is not None else None,
                         'fits': (answers.get('fits') or {}).get('noul'),
                         'jev_correct': (pick == target) if target is not None else None,
                         'identity_turn': llm_action['kind'] == 'tool' and llm_action['name'] in IDENTITY_TOOLS})
            reply = parse_message(message, tool_names_by_id)
            if message.tool_calls:
                call = message.tool_calls[0]
                tool_names_by_id[call.id] = (call.name, dict(call.arguments or {}))
            history.extend(reply)
        else:
            history.extend(parse_message(message, tool_names_by_id))
    return rows


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return {'p': p, 'low': centre - half, 'high': centre + half, 'n': n}


def rate(rows, key='jev_correct'):
    scored = [r for r in rows if r.get(key) is not None]
    return wilson(sum(1 for r in scored if r[key]), len(scored))


def report(rows):
    tool_rows = [r for r in rows if r['llm_action']['kind'] == 'tool']
    text_rows = [r for r in rows if r['llm_action']['kind'] == 'message']
    closed_rows = [r for r in rows if r['label_closed']]
    out = {'turns': len(rows), 'tool_turns': len(tool_rows), 'text_turns': len(text_rows), 'tasks': len({r['task_id'] for r in rows}),
           'X_closed_fraction': wilson(len(closed_rows), len(rows)),
           'X_closed_fraction_tool_turns_only': wilson(sum(1 for r in tool_rows if r['label_closed']), len(tool_rows)),
           'X_if_every_reviewed_text_turn_is_open': wilson(sum(1 for r in rows if r['label_closed'] and not r['needs_review']), len(rows)),
           'label_reasons': {}, 'compiler_flag_vs_label': {'closed&closed': 0, 'closed&open': 0, 'open&closed': 0, 'open&open': 0},
           'jev_agreement_with_llm': {'closed_turns': rate(closed_rows), 'tool_turns': rate(tool_rows), 'template_text_turns': rate([r for r in text_rows if r['label_closed']])},
           'positive_control_identity_turns': rate([r for r in rows if r['identity_turn']]),
           'by_confidence': {}, 'by_agreement': {}, 'by_cluster': {}, 'route_pick_vs_label': {}}
    for r in rows:
        if r['label_reason']:
            out['label_reasons'][r['label_reason'].split(':')[0]] = out['label_reasons'].get(r['label_reason'].split(':')[0], 0) + 1
        key = ('closed' if r['compiler_closed'] else 'open') + '&' + ('closed' if r['label_closed'] else 'open')
        out['compiler_flag_vs_label'][key] += 1
    bins = [('>=0.99', lambda c: c is not None and c >= 0.99), ('0.90-0.99', lambda c: c is not None and 0.90 <= c < 0.99), ('<0.90', lambda c: c is not None and c < 0.90)]
    for name, test in bins:
        out['by_confidence'][name] = rate([r for r in closed_rows if test(r['jev_confidence'])])
    for name, value in (('agree', True), ('disagree', False)):
        out['by_agreement'][name] = rate([r for r in closed_rows if r['agreement'] is value])
    for cluster in sorted({r['cluster'] for r in rows}):
        sub = [r for r in rows if r['cluster'] == cluster]
        out['by_cluster'][cluster] = {'turns': len(sub), 'closed_fraction': sum(1 for r in sub if r['label_closed']) / len(sub),
                                      'jev_agreement': rate([r for r in sub if r['label_closed']])}
    for r in rows:
        key = str(r['jev_route']) + '|' + ('closed' if r['label_closed'] else 'open')
        out['route_pick_vs_label'][key] = out['route_pick_vs_label'].get(key, 0) + 1
    x = out['X_closed_fraction']['p'] if out['X_closed_fraction'] else None
    pc = out['positive_control_identity_turns']
    verdicts = []
    if pc is None or pc['n'] == 0:
        verdicts.append('NO POSITIVE CONTROL: no identity-lookup turns in these traces; nothing below is read.')
    elif pc['p'] < POSITIVE_CONTROL_MIN:
        verdicts.append(f"POSITIVE CONTROL FAILED: Jev agrees with the LLM on {pc['p']:.0%} of identity-lookup turns (< {POSITIVE_CONTROL_MIN:.0%}); the state slice or the question is broken; nothing below is read.")
    if x is not None:
        if x < X_KILL:
            verdicts.append(f'KILL: X = {x:.0%} closed (< {X_KILL:.0%}).')
        else:
            verdicts.append(f"X = {x:.0%} closed [{out['X_closed_fraction']['low']:.0%}, {out['X_closed_fraction']['high']:.0%}] on {len(rows)} turns; predicted 60-70%.")
    j = out['jev_agreement_with_llm']['closed_turns']
    if j:
        verdicts.append(f"Jev agrees with the LLM on {j['p']:.0%} [{j['low']:.0%}, {j['high']:.0%}] of {j['n']} closed turns.")
    out['verdicts'] = verdicts
    return out


def render(rep):
    def w(x):
        return '—' if not x else f"{x['p']:.2f} [{x['low']:.2f}, {x['high']:.2f}] n={x['n']}"
    lines = ['# Stage 0: closed fraction and Jev replay', '', f"turns {rep['turns']} (tool {rep['tool_turns']}, text {rep['text_turns']}) over {rep['tasks']} tasks", '',
             '| quantity | value |', '|---|---|',
             f"| X, closed fraction, all turns | {w(rep['X_closed_fraction'])} |",
             f"| X, tool turns only | {w(rep['X_closed_fraction_tool_turns_only'])} |",
             f"| X if every unreviewed text label is open | {w(rep['X_if_every_reviewed_text_turn_is_open'])} |",
             f"| Jev = LLM on closed turns | {w(rep['jev_agreement_with_llm']['closed_turns'])} |",
             f"| Jev = LLM on tool turns | {w(rep['jev_agreement_with_llm']['tool_turns'])} |",
             f"| Jev = LLM on template text turns | {w(rep['jev_agreement_with_llm']['template_text_turns'])} |",
             f"| positive control, identity-lookup turns | {w(rep['positive_control_identity_turns'])} |", '',
             '| compiler flag & label | turns |', '|---|---|'] + [f'| {k} | {v} |' for k, v in rep['compiler_flag_vs_label'].items()] + \
            ['', '| label reason | turns |', '|---|---|'] + [f'| {k} | {v} |' for k, v in sorted(rep['label_reasons'].items())] + \
            ['', '| Jev confidence on closed turns | agreement |', '|---|---|'] + [f'| {k} | {w(v)} |' for k, v in rep['by_confidence'].items()] + \
            ['', '| paraphrase agreement on closed turns | agreement with LLM |', '|---|---|'] + [f'| {k} | {w(v)} |' for k, v in rep['by_agreement'].items()] + \
            ['', '| Jev route pick & label | turns |', '|---|---|'] + [f'| {k} | {v} |' for k, v in sorted(rep['route_pick_vs_label'].items())] + \
            ['', '| cluster | turns | closed | Jev = LLM on closed |', '|---|---|---|---|'] + \
            [f"| {k} | {v['turns']} | {v['closed_fraction']:.2f} | {w(v['jev_agreement'])} |" for k, v in rep['by_cluster'].items()] + \
            ['', 'Verdicts:'] + [f'- {v}' for v in rep['verdicts']]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', nargs='*', default=None, help='tau2 results.json files of LLM-only runs; default: runs/tau2/*/llm_only/results.json')
    p.add_argument('--domain', default='retail')
    p.add_argument('--jev-model', default=seeds.JEV_MODEL)
    p.add_argument('--namespace', default=None)
    p.add_argument('--mock', action='store_true', help='mock Jev (plumbing only; the replay numbers are meaningless)')
    p.add_argument('--out', default='runs/stage0')
    args = p.parse_args(argv)
    load_dotenv(ROOT / '.env', override=False)
    adapter.ensure_tau2()
    from tau2.data_model.simulation import Results
    from tau2.runner import build_environment
    from examples.tau2.agent import tool_schema
    paths = args.results or sorted(glob.glob(str(ROOT / 'runs/tau2/*/llm_only/results.json')))
    if not paths:
        p.error('no LLM-only results.json found')
    if not args.mock and not os.getenv('TYPESAFE_API_KEY'):
        p.error('TYPESAFE_API_KEY is not set')
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = ROOT / args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    jev = JevClient(mock=True) if args.mock else JevClient(model=args.jev_model, cache_dir=str(ROOT / '.cache/jev'), cache_namespace=args.namespace or f'stage0-{stamp}')
    env = build_environment(args.domain)
    tool_schemas = {tool.name: tool_schema(tool) for tool in env.get_tools()}
    policy = env.get_policy()
    runtime = Runtime(compile_spec(), jev, max_workers=1)
    rows = []
    try:
        for path in paths:
            results = Results.model_validate_json(Path(path).read_text())
            for sim in results.simulations or []:
                if sim.reward_info is None:
                    continue
                for row in walk(sim, tool_schemas, policy, jev, runtime):
                    row['source'] = str(path)
                    rows.append(row)
    finally:
        runtime.close()
    with (out_dir / 'labels.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ('source', 'task_id', 'turn', 'incoming_role', 'cluster', 'compiler_closed', 'options', 'needs', 'llm_action', 'label_closed', 'label_reason', 'target', 'needs_review')}, default=str) + '\n')
    with (out_dir / 'replay.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + '\n')
    rep = report(rows)
    rep['manifest'] = {'results': paths, 'jev_model': args.jev_model, 'mock': args.mock, 'jev_stats': jev.stats, 'jev_actual_models': jev.metadata['actual_models'], 'stamp': stamp}
    (out_dir / 'report.json').write_text(json.dumps(rep, indent=1, default=str))
    text = render(rep)
    (out_dir / 'report.md').write_text(text)
    print(text)
    print(f'artifacts: {out_dir}')


if __name__ == '__main__':
    main()
