"""Smallest real-model evolution run on a bundled example task.

Reads credentials from the environment or a local .env (never committed):
  TYPESAFE_API_KEY     Jev, direct TypeSafe API (https://console.typesafe.ai/keys)
  ANTHROPIC_API_KEY    llm and agent nodes; only needed for the llm seed
Reflection uses the local `claude` CLI (configs/claude_local.json); no key needed.

    uv run python scripts/live_run.py --task classify --seed jev --rounds 2 --train 6 --validation 6
    uv run python scripts/live_run.py --task constrained_text --seed llm --rounds 1 --train 4 --validation 4
"""
import argparse
import importlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from mixedharness.evolve import run_evolution
from mixedharness.jev import JevClient
from mixedharness.judge import make_judge, parse_judge_spec
from mixedharness.llm import LLMClient
from mixedharness.reflect import Proposer
from mixedharness.seeds import ROUTES, route_of_spec, seed_for_route
from mixedharness.store import RunStore

TASKS = {'classify': 'examples.classify.adapter', 'constrained_text': 'examples.constrained_text.adapter'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=sorted(TASKS), default='classify')
    parser.add_argument('--seed', choices=list(ROUTES), default=None, help='seed shape; default: jev for a labels/catalog task, llm otherwise')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--batch', type=int, default=2, help='training episodes per reflection')
    parser.add_argument('--train', type=int, default=6)
    parser.add_argument('--validation', type=int, default=6)
    parser.add_argument('--latency-ms', type=float, default=3000, help='p95 per-decision budget')
    parser.add_argument('--jev-model', default='jev-1.13.0')
    parser.add_argument('--judge', default=None, help='jev (default), jev:<model>, or logprob:<model>@<base_url>')
    parser.add_argument('--llm-model', default='claude-sonnet-5')
    parser.add_argument('--proposer-config', default='configs/claude_local.json')
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--namespace', default=None, help='response cache namespace; reuse to replay recorded calls')
    parser.add_argument('--mock', action='store_true', help='mock Jev, LLM and proposer (plumbing only)')
    args = parser.parse_args(argv)

    load_dotenv(Path.cwd() / '.env', override=False)
    judge_spec = args.judge or f'jev:{args.jev_model}'
    if not args.mock and parse_judge_spec(judge_spec)['kind'] == 'jev' and not (os.getenv('TYPESAFE_API_KEY') or os.getenv('TYPESAFE_API_KEY_PERSONAL') or os.getenv('TYPESAFE_API_KEY_personal')):
        parser.error('TYPESAFE_API_KEY is not set (put it in .env or the environment, or pass --judge logprob:<model>@<base_url>)')
    namespace = args.namespace or f'live-{args.task}'
    adapter = importlib.import_module(TASKS[args.task])
    task, train = adapter.load('train')
    _, validation = adapter.load('validation')
    if args.mock:
        jev, llm = JevClient(mock=True), LLMClient(kind='mock')
        proposer = Proposer({'kind': 'mock', 'responder': lambda prompt: json.dumps(json.loads(prompt)['parent_spec'])})
    else:
        jev = make_judge(judge_spec, cache_dir='.cache/jev', cache_namespace=namespace)
        llm = LLMClient(kind='anthropic', model=args.llm_model, cache_dir='.cache/llm', cache_namespace=namespace)
        proposer = Proposer(json.loads(Path(args.proposer_config).read_text()))
    seed = args.seed or ('jev' if task['action_space']['kind'] in ('labels', 'catalog') else 'llm')
    spec = seed_for_route(seed, task, jev_model=args.jev_model, llm_model=llm.model)
    spec['llm_model'] = llm.model   # pinned so a proposal may add llm nodes
    summary = run_evolution(train[:args.train], validation[:args.validation], jev=jev, llm=llm, proposer=proposer,
                            store=RunStore(args.runs), evaluator=adapter.evaluate_episode, seed_spec=spec,
                            task={'id': args.task, 'objective': task['objective'], 'seed': seed}, rounds=args.rounds,
                            batch_size=args.batch, budget={'latency_ms_p95': args.latency_ms}, run_name=f'{args.task}-{seed}')
    report = {key: summary.get(key) for key in ('run_id', 'seed_score', 'best_score', 'reflections', 'accepted', 'total_metric_calls')}
    report['seed'] = seed
    report['final_shape'] = route_of_spec(summary['best_spec'])
    report['jev'] = {'model': args.jev_model, 'actual_models': jev.metadata['actual_models'], **jev.stats}
    report['llm'] = llm.stats
    report['proposer'] = proposer.stats
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nInspect: {args.runs}/{summary['run_id']}/")


if __name__ == '__main__':
    main()
