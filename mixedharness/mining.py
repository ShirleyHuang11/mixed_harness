"""Loop A, macro-action mining: tool-call sequences the LLM path repeats become candidates for new Jev options.

    candidates = mine_macro_actions(steps, min_support=3, max_len=3)

`steps` are step-store rows. Within each episode, consecutive steps whose realised path was narrow or escalate
and whose action was a tool call form runs; every n-gram of tool names (1 <= n <= max_len) inside a run is
counted across episodes. A candidate is an n-gram with support (distinct episodes) >= min_support. The output is
data for a reflection model or a person: {sequence, support, episodes, share_of_llm_tool_steps}. Nothing is wired
into a harness automatically; promoting a candidate to an option means writing its compiler branch and argument
fill, which is what a proposal in the evolution loop does.
"""


def _runs(steps):
    by_episode = {}
    for s in steps:
        by_episode.setdefault((s.get('episode'), s.get('trial')), []).append(s)
    runs = []
    for key, rows in by_episode.items():
        rows = sorted(rows, key=lambda r: (r.get('turn') or 0))
        current = []
        for r in rows:
            llm_tool = r.get('path') in ('narrow', 'escalate') and r.get('action_kind') == 'tool' and r.get('action_name')
            if llm_tool:
                current.append(r['action_name'])
            elif current:
                runs.append((key, current)); current = []
        if current:
            runs.append((key, current))
    return runs


def mine_macro_actions(steps, *, min_support=3, max_len=3):
    runs = _runs(steps)
    total_llm_tool_steps = sum(len(seq) for _, seq in runs)
    support = {}
    for episode, seq in runs:
        seen = set()
        for n in range(1, max_len + 1):
            for i in range(len(seq) - n + 1):
                gram = tuple(seq[i:i + n])
                if gram not in seen:
                    seen.add(gram)
                    support.setdefault(gram, set()).add(episode)
    out = []
    for gram, episodes in support.items():
        if len(episodes) >= min_support:
            out.append({'sequence': list(gram), 'support': len(episodes), 'episodes': sorted(str(e[0]) for e in episodes),
                        'share_of_llm_tool_steps': (len(episodes) * len(gram) / total_llm_tool_steps) if total_llm_tool_steps else 0.0})
    out.sort(key=lambda c: (-c['support'], -len(c['sequence']), c['sequence']))
    return {'llm_tool_steps': total_llm_tool_steps, 'runs': len(runs), 'candidates': out}
