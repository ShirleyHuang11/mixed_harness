"""A minimal OpenAI-compatible chat endpoint over a transformers model that returns first-token log-probabilities.

    python scripts/judge_server.py --model Qwen/Qwen3-4B --port 8790

Then the harness can use the model as its judge with no API key:

    --judge logprob:Qwen/Qwen3-4B@http://127.0.0.1:8790/v1

Only what the log-probability judge needs is implemented: POST /v1/chat/completions with `max_tokens: 1` and
`logprobs: true` returns the top-k next tokens after the chat template, with their log-probabilities, plus the
prompt token count. Requests are served one at a time on one device. vLLM, llama.cpp's server, SGLang and LM Studio
expose the same fields and can replace this script for throughput.

Needs torch and transformers (`uv pip install torch transformers`); runs on CPU for models of about 1B parameters
and on any GPU for larger ones. Qwen3 chat templates get `enable_thinking=False` so the first token is the answer.
"""
import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Model:
    def __init__(self, name, device=None, dtype=None, top_k=20):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = name
        self.top_k = top_k
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        wanted = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}.get(dtype or '', None)
        if wanted is None:
            wanted = torch.bfloat16 if self.device == 'cuda' else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=wanted).to(self.device).eval()
        self.lock = threading.Lock()

    def prompt_ids(self, messages, template_kwargs):
        kwargs = {'tokenize': False, 'add_generation_prompt': True}
        kwargs.update(template_kwargs or {})
        try:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:   # a template that does not accept enable_thinking
            kwargs = {'tokenize': False, 'add_generation_prompt': True}
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids

    def next_token_logprobs(self, messages, template_kwargs=None):
        torch = self.torch
        ids = self.prompt_ids(messages, template_kwargs).to(self.device)
        with self.lock, torch.no_grad():
            logits = self.model(ids).logits[0, -1].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            top = torch.topk(logprobs, self.top_k)
        tokens = [self.tokenizer.decode([int(i)]) for i in top.indices]
        return [(tok, float(lp)) for tok, lp in zip(tokens, top.values)], int(ids.shape[1])


MODEL = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
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
            return self._send(200, {'object': 'list', 'data': [{'id': MODEL.name, 'object': 'model'}]})
        self._send(200, {'ok': True, 'model': MODEL.name, 'device': MODEL.device})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        try:
            req = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            return self._send(400, {'error': {'message': 'invalid JSON'}})
        if not self.path.rstrip('/').endswith('/chat/completions'):
            return self._send(404, {'error': {'message': 'only /v1/chat/completions'}})
        messages = req.get('messages') or []
        started = time.time()
        try:
            top, n_prompt = MODEL.next_token_logprobs(messages, req.get('chat_template_kwargs'))
        except Exception as exc:   # report, never crash the server
            return self._send(500, {'error': {'message': str(exc)[:300]}})
        token, lp = top[0]
        content = [{'token': token, 'logprob': lp, 'top_logprobs': [{'token': t, 'logprob': l} for t, l in top]}]
        self._send(200, {'id': 'chatcmpl-' + uuid.uuid4().hex[:16], 'object': 'chat.completion', 'created': int(started), 'model': MODEL.name,
                         'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': token}, 'logprobs': {'content': content}, 'finish_reason': 'length'}],
                         'usage': {'prompt_tokens': n_prompt, 'completion_tokens': 1, 'total_tokens': n_prompt + 1},
                         'x_server_seconds': round(time.time() - started, 3)})


def main(argv=None):
    global MODEL
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--port', type=int, default=8790)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--device', default=None)
    p.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default=None)
    p.add_argument('--top-k', type=int, default=20)
    args = p.parse_args(argv)
    MODEL = Model(args.model, args.device, args.dtype, args.top_k)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write(f'judge server: {args.model} on {MODEL.device}, http://{args.host}:{args.port}/v1\n')
    server.serve_forever()


if __name__ == '__main__':
    main()
