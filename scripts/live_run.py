"""Smallest real-model run of the routed harness on a bundled example task.

Reads credentials from the environment or a local .env (never committed):
  TYPESAFE_API_KEY     Jev directly from TypeSafe (https://console.typesafe.ai/keys)   -> --jev-transport typesafe
  AI_GATEWAY_API_KEY   Jev through Vercel AI Gateway                                    -> --jev-transport vercel
  ANTHROPIC_API_KEY    llm nodes; only needed for open_ended / hybrid seeds
Reflection uses the local `claude` CLI via configs/claude_local.json (no key needed).

    uv run python scripts/live_run.py --task classify --rounds 2 --train 6 --validation 6
    uv run python scripts/live_run.py --task constrained_text --rounds 1 --train 4 --validation 4
    uv run python scripts/live_run.py --task constrained_text --route hybrid
"""
import argparse
import importlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from auto_jev.llm import LLMClient
from auto_jev.providers import JevClient, make_proposer
from auto_jev.routed import run_routed_evolution
from auto_jev.storage import RunStore

TASKS = {'classify': 'examples.classify.adapter', 'constrained_text': 'examples.constrained_text.adapter'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', choices=sorted(TASKS), default='classify')
    parser.add_argument('--route', choices=['settled', 'open_ended', 'hybrid'], default=None, help='override the triage')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--batch', type=int, default=2, help='training episodes per reflection')
    parser.add_argument('--train', type=int, default=6)
    parser.add_argument('--validation', type=int, default=6)
    parser.add_argument('--latency-ms', type=float, default=3000, help='p95 per-decision budget')
    parser.add_argument('--jev-transport', choices=['auto', 'typesafe', 'vercel'], default='auto',
                        help='auto = vercel if AI_GATEWAY_API_KEY is set, else typesafe')
    parser.add_argument('--jev-model', default='typesafe-ai/jev',
                        help="gateway alias; on the direct transport it maps to jev-1.13.0, or pass e.g. jev-latest")
    parser.add_argument('--llm-model', default='claude-sonnet-5')
    parser.add_argument('--proposer-config', default='configs/claude_local.json')
    parser.add_argument('--runs', default='runs')
    parser.add_argument('--namespace', default=None, help='response cache namespace; reuse to replay recorded calls')
    args = parser.parse_args(argv)

    load_dotenv(Path.cwd() / '.env', override=False)
    transport = args.jev_transport
    if transport == 'auto':
        transport = 'vercel' if os.getenv('AI_GATEWAY_API_KEY') else 'typesafe'
    key_env = 'AI_GATEWAY_API_KEY' if transport == 'vercel' else 'TYPESAFE_API_KEY'
    if not os.getenv(key_env):
        parser.error(f'{key_env} is not set for --jev-transport {transport} (put it in .env or the environment)')
    namespace = args.namespace or f'live-{args.task}'
    adapter = importlib.import_module(TASKS[args.task])
    task, train = adapter.load('train')
    _, validation = adapter.load('validation')

    jev = JevClient(transport=transport, model=args.jev_model, cache_dir='.cache/jev', cache_namespace=namespace)
    llm = LLMClient(kind='anthropic', model=args.llm_model, cache_dir='.cache/llm', cache_namespace=namespace)
    proposer = make_proposer(json.loads(Path(args.proposer_config).read_text()))

    summary = run_routed_evolution(
        task, train[:args.train], validation[:args.validation],
        jev=jev, llm=llm, proposer=proposer, store=RunStore(args.runs),
        evaluator=adapter.evaluate_episode, task_id=args.task, route=args.route,
        evolution_rounds=args.rounds, reflection_batch_size=args.batch,
        budget={'latency_ms_p95': args.latency_ms},
    )
    report = {key: summary.get(key) for key in (
        'routed_id', 'run_id', 'route', 'route_source', 'seed_route', 'final_route',
        'best_score', 'rounds_completed', 'rounds_accepted', 'total_metric_calls', 'questions_version')}
    report['triage_probabilities'] = summary['decision']['probabilities']
    report['triage_improvement'] = summary['triage_improvement']['status']
    report['jev'] = {'transport': transport, 'requested_model': args.jev_model, 'actual_model': jev.metadata.get('actual_model'), **jev.stats}
    report['llm'] = llm.stats
    report['proposer'] = getattr(proposer, 'stats', {})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nInspect: {args.runs}/routed/{summary['routed_id']}/  and  {args.runs}/{summary['run_id']}/")


if __name__ == '__main__':
    main()
