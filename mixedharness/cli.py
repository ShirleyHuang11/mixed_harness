"""Command line: `mixedharness doctor` reports what this machine can run; `mixedharness schema` prints the spec schema."""
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
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        print(json.dumps(doctor(), indent=1))
    elif args.command == 'schema':
        print(json.dumps(authoring_schema(), indent=1))


if __name__ == '__main__':
    main()
