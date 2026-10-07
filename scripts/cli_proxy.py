"""A local OpenAI-compatible chat endpoint backed by a subscription CLI (`codex exec` or `claude -p`).

    ~/tau2-bench/.venv/bin/python scripts/cli_proxy.py --backend codex --model gpt-5.5 --port 8765

Then any litellm/OpenAI client can use it:  model "openai/gpt-5.5", api_base "http://127.0.0.1:8765/v1", any api_key.
tau2's user simulator and NL-assertion judge, and its stock LLM agent, go through here when scripts/tau2_compare.py
is run with --llm-backend codex. The harness's own LLM nodes use LLMClient(kind='codex') directly, no proxy.

What it does with a request: flattens system + messages into one prompt; when `tools` are present it appends the
tool schemas and asks for exactly one JSON object, either {"tool": name, "arguments": {...}} or {"message": text},
and converts the first form into an OpenAI `tool_calls` entry. Token usage is estimated from characters (the CLI
reports none) and cost is reported as 0: the account is a subscription. Calls are serialised per worker slot;
run with --workers 3 to allow three CLI processes at once.

Limits: each call is a process start, 5 to 15 seconds; the subscription's usage window can refuse calls, which
surfaces as HTTP 503 and is retried by litellm. No streaming. Not an API replacement for production.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mixedharness.llm import extract_json  # noqa: E402

TOOL_INSTRUCTION = (
    "\n\nYou may call exactly one tool or send one message. Reply with exactly one JSON object and nothing else: "
    'either {"tool": "<tool name>", "arguments": {...}} with every required argument, or {"message": "<text to the user>"}. '
    "Available tools (JSON Schema):\n"
)


def flatten(messages, tools=None):
    parts = []
    for m in messages:
        role = m.get('role')
        content = m.get('content')
        if isinstance(content, list):
            content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
        if role == 'system':
            parts.append(f"<system>\n{content}\n</system>")
        elif role == 'tool':
            parts.append(f"<tool_result name={m.get('name')}>\n{content}\n</tool_result>")
        elif role == 'assistant' and m.get('tool_calls'):
            calls = [f"{c['function']['name']}({c['function'].get('arguments', '{}')})" for c in m['tool_calls']]
            parts.append(f"<assistant>[tool call] {'; '.join(calls)}</assistant>")
        else:
            parts.append(f"<{role}>\n{content}\n</{role}>")
    prompt = '\n'.join(parts)
    if tools:
        prompt += TOOL_INSTRUCTION + json.dumps([t.get('function', t) for t in tools], ensure_ascii=False)
    else:
        prompt += "\n\nReply as the assistant with the next message only; no preamble, no analysis."
    return prompt


class CLIBackend:
    def __init__(self, backend, model, timeout=180):
        self.backend, self.model, self.timeout = backend, model, timeout
        self.workdir = Path(tempfile.mkdtemp(prefix='cli-proxy-'))
        self.lock = threading.Semaphore(int(os.environ.get('CLI_PROXY_WORKERS', '3')))

    def complete(self, prompt):
        with self.lock:
            if self.backend == 'codex':
                last = ''
                for attempt in range(4):   # the CLI sometimes fails to start under concurrency; retry with backoff
                    out = self.workdir / (uuid.uuid4().hex + '.txt')
                    cmd = ['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '-s', 'read-only', '-C', str(self.workdir),
                           '-m', self.model, '-o', str(out), '-']
                    proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=self.timeout)
                    text = out.read_text().strip() if out.exists() else ''
                    if out.exists():
                        out.unlink()
                    if text:
                        return text
                    last = proc.stderr[-400:]
                    time.sleep(2 + 3 * attempt)
                raise RuntimeError(f'codex exec returned nothing after 4 attempts: {last}')
            cmd = ['claude', '-p', '--model', self.model, '--output-format', 'text', '--tools', '', '--no-session-persistence']
            proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=self.timeout, cwd=self.workdir)
            if proc.returncode or not proc.stdout.strip():
                raise RuntimeError(f'claude -p failed: {proc.stderr[-400:]}')
            return proc.stdout.strip()


BACKEND = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):   # one line per call
        sys.stderr.write('%s %s\n' % (time.strftime('%H:%M:%S'), fmt % args))

    def _send(self, code, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip('/').endswith('/models'):
            return self._send(200, {'object': 'list', 'data': [{'id': BACKEND.model, 'object': 'model'}]})
        self._send(200, {'ok': True, 'backend': BACKEND.backend, 'model': BACKEND.model})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        try:
            req = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            return self._send(400, {'error': {'message': 'invalid JSON'}})
        if not self.path.rstrip('/').endswith('/chat/completions'):
            return self._send(404, {'error': {'message': 'only /v1/chat/completions'}})
        messages, tools = req.get('messages') or [], req.get('tools') or None
        prompt = flatten(messages, tools)
        started = time.time()
        try:
            text = BACKEND.complete(prompt)
        except subprocess.TimeoutExpired:
            return self._send(504, {'error': {'message': 'cli timeout'}})
        except Exception as exc:
            return self._send(503, {'error': {'message': str(exc)[:300]}})
        message = {'role': 'assistant', 'content': text}
        finish = 'stop'
        if tools:
            try:
                obj = json.loads(extract_json(text))
            except Exception:
                obj = {'message': text}
            if isinstance(obj, dict) and obj.get('tool'):
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_' + uuid.uuid4().hex[:12], 'type': 'function',
                           'function': {'name': str(obj['tool']), 'arguments': json.dumps(obj.get('arguments') or {}, ensure_ascii=False)}}]}
                finish = 'tool_calls'
            else:
                message = {'role': 'assistant', 'content': str(obj.get('message', text)) if isinstance(obj, dict) else text}
        usage = {'prompt_tokens': len(prompt) // 4, 'completion_tokens': len(text) // 4, 'total_tokens': (len(prompt) + len(text)) // 4}
        self._send(200, {'id': 'chatcmpl-' + uuid.uuid4().hex[:16], 'object': 'chat.completion', 'created': int(started),
                         'model': req.get('model') or BACKEND.model,
                         'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': usage,
                         'x_cli_seconds': round(time.time() - started, 1)})


def main(argv=None):
    global BACKEND
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backend', choices=['codex', 'claude'], default='codex')
    p.add_argument('--model', default='gpt-5.5')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--timeout', type=int, default=180)
    args = p.parse_args(argv)
    os.environ['CLI_PROXY_WORKERS'] = str(args.workers)
    BACKEND = CLIBackend(args.backend, args.model, args.timeout)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    sys.stderr.write(f'cli proxy: {args.backend} {args.model} on http://127.0.0.1:{args.port}/v1 ({args.workers} workers)\n')
    server.serve_forever()


if __name__ == '__main__':
    main()
