"""Command line: `mixedharness doctor` reports what this machine can run; `mixedharness schema` prints the spec schema;
`mixedharness audit <traces>` measures how many of a recorded agent's turns a judge could take and what that would save."""
import argparse
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .sandbox import runtime_identity
from .spec import authoring_schema


def doctor():
    load_dotenv(Path.cwd() / '.env', override=False)
    return {'version': __version__,
            'sandbox': runtime_identity(),
            'claude_cli': shutil.which('claude') is not None,
            'credentials_present': {k: bool(os.getenv(k)) for k in ('TYPESAFE_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY')}}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='mixedharness')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='sandbox availability and which credentials are present (never their values)')
    sub.add_parser('schema', help='print the spec schema given to authoring and reflection models')
    audit = sub.add_parser('audit', help='the ruler: share of a recorded agent\'s turns that are closed choices a judge could take, and the cost that would save')
    audit.add_argument('paths', nargs='+', help='tau2 results.json files, or JSON lines of OpenAI-style conversations')
    audit.add_argument('--format', choices=['tau2', 'messages'], default='tau2')
    audit.add_argument('--judge', default='jev', help='jev (TypeSafe key), jev:<model>, logprob:<model>@<base_url> (any model served with log-probabilities), or mock')
    audit.add_argument('--judge-temperature', type=float, default=1.0, help='temperature on a logprob judge, from scripts/calibrate_judge.py (judge.json)')
    audit.add_argument('--tools', default=None, help='JSON file {tool name: description}; otherwise tools are read from the traces')
    audit.add_argument('--price-model', default=None, help='price turns from their token usage at this model\'s list price when the trace carries no cost (e.g. gpt-5.5)')
    audit.add_argument('--policy-text', default=None, help='a file whose text is shown to the judge as the agent\'s policy')
    audit.add_argument('--limit', type=int, default=None, help='only the first N conversations')
    audit.add_argument('--workers', type=int, default=4)
    audit.add_argument('--namespace', default=None, help='judge response cache namespace; reuse one to re-render an audit without new judge calls')
    audit.add_argument('--out', default=None, help='directory for audit.jsonl, report.json and report.md; default runs/audit/<stamp>')
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        print(json.dumps(doctor(), indent=1))
    elif args.command == 'schema':
        print(json.dumps(authoring_schema(), indent=1))
    elif args.command == 'audit':
        import datetime as dt
        from .audit import render, run_audit
        load_dotenv(Path.cwd() / '.env', override=False)
        out = Path(args.out) if args.out else Path('runs/audit') / dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        tools = json.loads(Path(args.tools).read_text()) if args.tools else None
        policy_text = Path(args.policy_text).read_text() if args.policy_text else None
        report, _ = run_audit(args.paths, fmt=args.format, judge_spec=args.judge, tools=tools, price_model=args.price_model, out_dir=out,
                              workers=args.workers, limit=args.limit, cache_dir=str(Path('.cache/jev')), policy_text=policy_text, namespace=args.namespace,
                              judge_temperature=args.judge_temperature)
        print(render(report))
        print(f'artifacts: {out}')


if __name__ == '__main__':
    main()
