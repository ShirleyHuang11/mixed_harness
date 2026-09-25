"""Reflection: turn complete training traces into a proposal for a rewritten spec.

`build_prompt(...)` packs the task context, the spec format, the parent spec and the complete traces of
the training batch into one JSON payload with an explicit byte cap. Over the cap, the payload is archived
and refused; nothing is truncated silently, because a reflection model that saw half a trace would propose
from half the evidence.

`Proposer` runs the reflection model. Three kinds: `claude_cli` (the local `claude` command with tools
disabled; no key needed), `anthropic`, `openai`. It returns the model's text; parsing and validation of
the proposed spec happen in `evolve.py`.
"""
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time

from .spec import authoring_schema

DEFAULT_MAX_PROMPT_BYTES = 1_000_000

INSTRUCTION = (
    'You are improving a decision harness: a directed acyclic graph of python, jev, llm and expression nodes '
    'that turns an observation into an action. Read the parent spec and ALL of the training trajectories '
    'below, including every node input and output, every Jev question and answer, every LLM prompt and reply, '
    'and every failure. Then return ONLY a complete, valid spec JSON object (no prose, no code fences) that '
    'should score higher on this task. You may add, remove or rewire nodes, change a node\'s kind, move logic '
    'between code and model calls, rewrite Jev questions and criteria, and change output and memory rules. '
    'Keep format, jev_model and llm_model exactly as in the parent. Python nodes must declare every node they '
    'read in inputs. Training outcomes are hindsight feedback, never runtime observations. Stay within the '
    'stated budget: a candidate over budget is invalid.'
)


class ReflectionError(RuntimeError):
    """The reflection model could not be run, or its input exceeded the cap. Infrastructure."""


def build_prompt(task, parent_spec, trajectories, *, budget=None, max_bytes=DEFAULT_MAX_PROMPT_BYTES):
    """Return (prompt_text, payload). Raises ReflectionError when the payload exceeds max_bytes."""
    payload = {'instruction': INSTRUCTION, 'task': task, 'budget': budget, 'spec_schema': authoring_schema(),
               'parent_spec': parent_spec, 'training_trajectories': trajectories}
    text = json.dumps(payload, ensure_ascii=False, indent=None, default=str)
    size = len(text.encode('utf-8'))
    payload['bytes'] = size
    if size > max_bytes:
        raise ReflectionError(f'reflection input is {size} bytes; the cap is {max_bytes}')
    return text, payload


def strip_fences(text):
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[-1].strip().startswith('```'):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = '\n'.join(lines).strip()
    return text


class Proposer:
    """Callable: prompt text -> model text. `stats` and `metadata` describe what actually ran."""

    def __init__(self, config):
        self.config = dict(config)
        self.kind = self.config.get('kind', 'claude_cli')
        if self.kind not in ('claude_cli', 'anthropic', 'openai', 'mock'):
            raise ValueError(f'unknown proposer kind {self.kind!r}')
        self.stats = {'calls': 0, 'elapsed_ms': 0.0, 'input_tokens': 0, 'output_tokens': 0, 'reported_cost_usd': 0.0}
        self.metadata = {k: v for k, v in self.config.items() if k not in ('api_key',)}
        self.metadata['actual_models'] = []

    def __call__(self, prompt):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('prompt must be a nonempty string')
        started = time.perf_counter()
        if self.kind == 'mock':
            text = self.config['responder'](prompt)
        elif self.kind == 'claude_cli':
            text = self._claude_cli(prompt)
        else:
            text = self._api(prompt)
        if not isinstance(text, str) or not text.strip():
            raise ReflectionError('empty reflection reply')
        self.stats['calls'] += 1
        self.stats['elapsed_ms'] += (time.perf_counter() - started) * 1000
        return text

    def _claude_cli(self, prompt):
        executable = self.config.get('executable') or shutil.which('claude')
        if not executable:
            raise ReflectionError('the claude CLI is not installed')
        model = self.config.get('model', 'claude-fable-5-1')
        args = [executable, '--print', '--model', model, '--effort', self.config.get('effort', 'xhigh'), '--safe-mode',
                '--tools', '', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}', '--disable-slash-commands',
                '--no-session-persistence', '--output-format', 'json',
                '--system-prompt', 'Return only the requested complete spec JSON object. Use only the provided observations; no tools.']
        env = {k: v for k, v in os.environ.items() if not k.endswith('_API_KEY')}
        with tempfile.TemporaryDirectory(prefix='mixedharness-reflect-') as folder:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, cwd=folder, env=env, start_new_session=True)
            try:
                out, err = process.communicate(prompt, timeout=self.config.get('timeout', 600))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()
                raise ReflectionError('the reflection model timed out') from None
        try:
            envelope = json.loads(out)
        except ValueError:
            raise ReflectionError(f'claude returned an invalid envelope: {err[-300:]}') from None
        if process.returncode or envelope.get('is_error'):
            raise ReflectionError(f'claude reflection call failed: {str(envelope.get("result"))[:300]}')
        usage = envelope.get('modelUsage') or {}
        self.metadata['actual_models'] = sorted(set(self.metadata['actual_models']) | set(usage))
        if model not in usage:
            raise ReflectionError('claude did not confirm the requested model')
        self.stats['input_tokens'] += sum((u.get('inputTokens', 0) + u.get('cacheReadInputTokens', 0) + u.get('cacheCreationInputTokens', 0)) for u in usage.values())
        self.stats['output_tokens'] += sum(u.get('outputTokens', 0) for u in usage.values())
        self.stats['reported_cost_usd'] += sum(u.get('costUSD', 0.0) for u in usage.values())
        return envelope.get('result', '')

    def _api(self, prompt):
        key_env = self.config.get('api_key_env') or ('ANTHROPIC_API_KEY' if self.kind == 'anthropic' else 'OPENAI_API_KEY')
        key = os.getenv(key_env)
        if not key:
            raise ReflectionError(f'{key_env} is not set')
        model = self.config['model']
        try:
            if self.kind == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=key, timeout=self.config.get('timeout', 300))
                with client:
                    response = client.messages.create(model=model, max_tokens=self.config.get('max_tokens', 8192),
                                                      messages=[{'role': 'user', 'content': prompt}])
                text = ''.join(b.text for b in response.content if getattr(b, 'type', '') == 'text')
                self.stats['input_tokens'] += response.usage.input_tokens
                self.stats['output_tokens'] += response.usage.output_tokens
                actual = response.model
            else:
                import openai
                client = openai.OpenAI(api_key=key, base_url=self.config.get('base_url'), timeout=self.config.get('timeout', 300))
                with client:
                    response = client.chat.completions.create(model=model, max_completion_tokens=self.config.get('max_tokens', 8192),
                                                              messages=[{'role': 'user', 'content': prompt}])
                text = response.choices[0].message.content or ''
                if response.usage:
                    self.stats['input_tokens'] += response.usage.prompt_tokens
                    self.stats['output_tokens'] += response.usage.completion_tokens
                actual = response.model
        except ReflectionError:
            raise
        except Exception as exc:
            raise ReflectionError(f'reflection call failed: {type(exc).__name__}: {str(exc)[:200]}') from None
        self.metadata['actual_models'] = sorted(set(self.metadata['actual_models']) | {str(actual)})
        return text
