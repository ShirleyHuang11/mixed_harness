# Routed Harness Level 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fork JevHarness into this repo and add a Level-1 routed harness: Linux sandbox, `llm` nodes, budgets, Jev-based triage, per-route seed templates, `run_routed_evolution` that seeds one harness from the chosen route and runs the existing evolution once, and a replay-validated improvement step for the triage criteria after every run.

**Architecture:** `auto_jev` stays the runtime/evolution core. New modules are thin and single-purpose: `schema.py` (JSON-schema subset), `llm.py` (LLM transport, mirrors `JevClient`), `budget.py` (usage + budget checks), `triage.py` (probe → Jev triage → seed templates), `triage_store.py` (persisted triage questions, history, replay-validated improvement), `routed.py` (triage → one seed → one `run_evolution` → record → improve triage). Spec version 4 adds `llm` nodes. Examples live under `examples/` with fixtures so everything runs with mock providers.

**Tech Stack:** Python 3.11+ via `uv`; `gepa==0.1.4`, `httpx`, `anthropic`, `openai` (already in upstream `pyproject.toml`); `bwrap` (bubblewrap) on Linux, `sandbox-exec` on macOS; `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-22-routed-harness-level1-design.md`

## Global Constraints

- Python `>=3.11`; run everything through `uv run ...`. Do not add dependencies beyond upstream `pyproject.toml`.
- Functional Python nodes never execute on the host. Unavailable isolation raises `PythonSandboxUnavailable` (has `task_infrastructure_error = True`).
- Every new module is importable without API keys; `mock` providers must exercise every code path used by tests.
- Never send an explicit key to the wrong transport; never log key values. Keys are read only from `os.getenv(<api_key_env>)`.
- Spec versions 1–3 keep their exact current behavior; version 4 is additive.
- All JSON written by the system is finite (`allow_nan=False`) and passes `spec._json`.
- Commit after every task with a conventional message (`feat:`, `test:`, `chore:`, `docs:`). Do not commit `runs/`, `.cache/`, `.venv/`.

## Review Focus

1. A candidate spec whose `llm` node returns JSON that parses but fails `output_schema` must produce a candidate error (node `status: error`), never a crash or a silently-accepted value — pinned in Task 4 (`test_llm_node_schema_failure_is_a_node_error`).
2. Triage with a Jev answer whose `choice` key is not in the criteria (a malformed provider response) must raise `ValueError`, not pick a route — pinned in Task 6 (`test_triage_rejects_unknown_route`).
3. A budget of `{"latency_ms_p95": 0}` must not turn an *infrastructure* failure into `over_budget`; over-budget is only applied to successfully scored episodes — pinned in Task 5 (`test_budget_not_applied_to_failed_episode`).
4. An evaluator that does not accept an `llm` kwarg must still work with `run_evolution(..., llm=...)` — pinned in Task 5 (`test_legacy_evaluator_without_llm_param`).
5. On Linux with `bwrap` present but user namespaces disabled, `evaluate_python` must raise `PythonSandboxUnavailable`, not a candidate `ValueError` — pinned in Task 1 (`test_bwrap_startup_failure_is_infrastructure_error`).

---

### Task 0: Vendor upstream JevHarness and establish a baseline

**Files:**
- Create: `auto_jev/**` (copied), `tests/**` (copied subset), `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, `configs/*.json` (minus `azure_gateway.json`), `docs/task-authoring.md`, `docs/jev-example.md`, `docs/examples/*`
- Create: `tests/support.py`

**Interfaces:**
- Produces: the upstream `auto_jev` package at `/u/shuang24/mixed_harness/auto_jev`, `tests.support.ScriptedJev`.

- [ ] **Step 1: Copy the upstream checkout (already cloned at the scratchpad path)**

```bash
SRC=/tmp/claude-95970/-u-shuang24-mixed-harness/345c97e6-6ee5-49df-a11a-f81b7d5e5aa3/scratchpad/JevHarness
cd /u/shuang24/mixed_harness
cp -r $SRC/auto_jev ./auto_jev
mkdir -p tests docs/examples configs
cp $SRC/tests/test_datasets.py $SRC/tests/test_engine.py $SRC/tests/test_episode_trace.py \
   $SRC/tests/test_evaluation_history.py $SRC/tests/test_evolution_rounds.py $SRC/tests/test_experiment.py \
   $SRC/tests/test_experiment_recovery.py $SRC/tests/test_flow.py $SRC/tests/test_news.py $SRC/tests/test_paper.py \
   $SRC/tests/test_pipeline_v3.py $SRC/tests/test_proposers.py $SRC/tests/test_provider_concurrency.py \
   $SRC/tests/test_python_nodes.py $SRC/tests/test_python_runtime_identity.py $SRC/tests/test_reflection.py \
   $SRC/tests/test_review.py $SRC/tests/test_review_storage.py $SRC/tests/test_server.py \
   $SRC/tests/test_task_contract.py $SRC/tests/test_trace_codec.py $SRC/tests/test_trace_paging.py \
   $SRC/tests/test_vercel.py tests/
cp $SRC/pyproject.toml $SRC/uv.lock $SRC/.env.example $SRC/.gitignore .
cp $SRC/configs/anthropic.json $SRC/configs/claude_local.json $SRC/configs/claude_scale.json \
   $SRC/configs/flow_seed.json $SRC/configs/openai.json configs/
cp $SRC/docs/task-authoring.md $SRC/docs/jev-example.md docs/
cp $SRC/docs/examples/jev_call.py $SRC/docs/examples/pokemon-turn12-jev.json docs/examples/
```

Deliberately not copied: `examples/pokemon/`, `website/`, `skills/`, `scripts/`, `.claude-plugin/`, Pokémon/site/skill tests, `configs/azure_gateway.json` (internal endpoint), `configs/*_1000_50.json`, `configs/opus5_pokemon.json`.

- [ ] **Step 2: Remove the hard-coded internal Azure endpoint default**

In `auto_jev/providers.py`, in the `kind=='azure'` branch, replace
`azure_endpoint=cfg.get('azure_endpoint','https://aidp.bytedance.net/api/modelhub/online/v2/crawl')`
with `azure_endpoint=cfg['azure_endpoint']` (a missing endpoint is a configuration error, not a default).

- [ ] **Step 3: Rename the package description and README**

Edit `pyproject.toml`: `description = "Routed Jev/LLM decision harnesses with GEPA evolution"`. Overwrite `README.md`:

```markdown
# mixed_harness

Fork of [JevHarness](https://github.com/TianyuCodings/JevHarness) (`auto_jev`) extended with a routing layer:
a task is probed and triaged (settled / open-ended / hybrid), one harness is seeded per plausible route,
each is evolved with GEPA under the same budget, and the best validation score wins.

- Design: `docs/superpowers/specs/2026-09-22-routed-harness-level1-design.md`
- Diagram: https://claude.ai/artifact/PDRJpw5s2nNLd4opTtxqmA

## Setup

    uv sync --locked --extra dev
    uv run pytest

Functional Python nodes run under `bwrap` on Linux and `sandbox-exec` on macOS.
```

- [ ] **Step 4: Create the shared test helper**

Create `tests/support.py`:

```python
"""Scripted providers for tests. No network, deterministic, JevClient-shaped responses."""
import copy


class ScriptedJev:
    """Answer questions from a script: {question_id: {option: prob}} for choice/score, float for noul.

    Unscripted questions fall back to the first option / 0.5. Records every request.
    """
    mock = True
    transport = 'mock'

    def __init__(self, script=None):
        self.script = copy.deepcopy(script or {})
        self.requests = []
        self.stats = {'calls': 0, 'cache_hits': 0, 'input_tokens': 0, 'output_tokens': 0,
                      'elapsed_ms': 0.0, 'cost_usd': 0.0, 'unpriced_calls': 0}
        self.metadata = {'transport': 'mock', 'requested_model': 'typesafe-ai/jev',
                         'model_version_pinned': False, 'cache_namespace': 'scripted', 'cache_dir': None}

    def judge(self, state, questions, model=None):
        self.requests.append({'state': copy.deepcopy(state), 'questions': copy.deepcopy(questions)})
        self.stats['calls'] += 1
        answers = {}
        for qid, question in questions.items():
            scripted = self.script.get(qid)
            kind = question['type']
            if kind == 'noul':
                answers[qid] = {'type': 'noul', 'noul': float(scripted) if scripted is not None else 0.5}
            elif kind == 'choice':
                options = list(question['criteria'])
                probs = dict(scripted) if isinstance(scripted, dict) else {o: (1.0 if i == 0 else 0.0) for i, o in enumerate(options)}
                choice = max(probs, key=probs.get)
                answers[qid] = {'type': 'choice', 'choice': choice, 'probabilities': probs, 'confidence': probs[choice]}
            else:
                levels = len(question['criteria'])
                probs = dict(scripted) if isinstance(scripted, dict) else {str(i): (1.0 if i == 0 else 0.0) for i in range(levels)}
                score = sum(int(k) * v for k, v in probs.items())
                answers[qid] = {'type': 'score', 'score': score, 'probabilities': probs, 'confidence': max(probs.values())}
        return {'answers': answers, 'model': 'scripted', 'usage': {}, 'cache_hit': False,
                'elapsed_ms': 1.0, 'client_metadata': copy.deepcopy(self.metadata)}
```

- [ ] **Step 5: Install and run the baseline**

Run: `cd /u/shuang24/mixed_harness && uv sync --locked --extra dev && uv run pytest -q 2>&1 | tail -30`

Expected: most tests pass. On Linux the following fail *by design until Task 1*: `tests/test_python_nodes.py` (sandbox unavailable), `tests/test_python_runtime_identity.py` (darwin fixture), `tests/test_pipeline_v3.py` (python nodes), `tests/test_task_contract.py` if it executes python nodes. Record the exact failing test names in the commit message.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: vendor JevHarness auto_jev core and tests

Baseline on Linux: <paste failing test ids>"
```

---

### Task 1: Linux sandbox via bwrap

**Files:**
- Modify: `auto_jev/python_nodes.py` (`_sandbox_command`, new `_command_layout`, `runtime_identity`, `_BOOTSTRAP`, `_check_memory`, `evaluate_python` failure detection)
- Modify: `tests/test_python_nodes.py` (mark darwin-only tests), `tests/test_python_runtime_identity.py` (mark darwin-fixture tests)
- Create: `tests/test_python_nodes_linux.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `python_nodes._sandbox_command() -> list[str]` (darwin: unchanged layout; linux: `['/usr/bin/bwrap', *args, '--', <python>, '-I', '-S', '-B']`), `python_nodes._command_layout(command) -> dict(executable, sandbox_executable, profile, flags)`, `python_nodes._check_memory_darwin`, `python_nodes._check_memory_linux`.

- [ ] **Step 1: Write the failing Linux tests**

Create `tests/test_python_nodes_linux.py`:

```python
"""bwrap isolation on Linux. Skipped elsewhere."""
import copy
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

import auto_jev.python_nodes as python_nodes
from auto_jev.python_nodes import PythonSandboxUnavailable, evaluate_python

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='bwrap path is Linux only')

CONTEXT = {'obs': {'values': [1, 2, 3]}, 'nodes': {}, 'memory': {}}


def source(body):
    return 'def run(obs, nodes, memory):\n' + '\n'.join('    ' + line for line in body.splitlines()) + '\n'


def bwrap_available():
    try:
        python_nodes._sandbox_command()
    except PythonSandboxUnavailable:
        return False
    return True


def test_linux_command_uses_bwrap_with_isolation_flags():
    if not shutil.which('bwrap'):
        pytest.skip('bwrap not installed')
    command = python_nodes._sandbox_command()
    assert command[0].endswith('bwrap')
    for flag in ('--unshare-all', '--die-with-parent', '--new-session', '--clearenv', '--proc', '--dev', '--tmpfs'):
        assert flag in command
    separator = command.index('--')
    assert command[separator + 2:] == ['-I', '-S', '-B']
    layout = python_nodes._command_layout(command)
    assert layout['executable'] == command[separator + 1]
    assert layout['sandbox_executable'] == command[0]
    assert layout['flags'] == ['-I', '-S', '-B']
    assert json.loads(layout['profile']) == command[1:separator]


def test_missing_bwrap_fails_closed(monkeypatch):
    monkeypatch.setattr(python_nodes.shutil, 'which', lambda name: None)
    with pytest.raises(PythonSandboxUnavailable):
        python_nodes._sandbox_command()


def test_bwrap_startup_failure_is_infrastructure_error(monkeypatch):
    """bwrap present, namespaces refused: output empty, stderr names bwrap, exit 1."""
    process = SimpleNamespace(stdin=SimpleNamespace(close=lambda: None, closed=True),
                              stdout=SimpleNamespace(close=lambda: None, closed=True),
                              stderr=SimpleNamespace(close=lambda: None, closed=True),
                              returncode=1, poll=lambda: 1, wait=lambda timeout=None: 1, pid=1)
    monkeypatch.setattr(python_nodes, '_sandbox_command', lambda: ['/usr/bin/bwrap', '--', 'python', '-I', '-S', '-B'])
    monkeypatch.setattr(python_nodes.subprocess, 'Popen', lambda *a, **kw: process)
    monkeypatch.setattr(python_nodes, '_exchange', lambda *a: (b'', b'bwrap: No permissions to create a new namespace\n'))
    monkeypatch.setattr(python_nodes, '_kill', lambda p: None)
    with pytest.raises(PythonSandboxUnavailable):
        evaluate_python(source('return 1'), copy.deepcopy(CONTEXT))


@pytest.mark.skipif(not bwrap_available(), reason='bwrap unavailable')
def test_real_bwrap_runs_functional_python():
    assert evaluate_python(source('return sum(obs["values"])'), copy.deepcopy(CONTEXT)) == 6


@pytest.mark.skipif(not bwrap_available(), reason='bwrap unavailable')
def test_real_bwrap_enforces_address_space_limit():
    with pytest.raises(ValueError) as caught:
        evaluate_python(source('big = bytearray(700 * 1024 * 1024)\nreturn len(big)'), copy.deepcopy(CONTEXT))
    assert 'MemoryError' in str(caught.value)
    assert not getattr(caught.value, 'task_infrastructure_error', False)


@pytest.mark.skipif(not bwrap_available(), reason='bwrap unavailable')
def test_real_bwrap_denies_host_files_network_fork_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTO_JEV_SANDBOX_TEST_MARKER', 'synthetic-value')
    canary = tmp_path / 'synthetic-canary.txt'
    canary.write_text('synthetic-value')
    writable = tmp_path / 'must-not-be-written.txt'
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    probe = '''import json, os, socket
results = {}
def attempt(label, fn):
    try:
        fn()
    except OSError as exc:
        results[label] = "denied"
    else:
        results[label] = "ALLOWED"
attempt("read", lambda: open(CANARY).read())
attempt("write", lambda: open(WRITABLE, "w").write("x"))
attempt("network", lambda: socket.create_connection(("127.0.0.1", PORT), timeout=1))
def fork():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
attempt("fork", fork)
results["environment"] = "clean" if "AUTO_JEV_SANDBOX_TEST_MARKER" not in os.environ else "inherited"
print(json.dumps(results))
'''.replace('CANARY', repr(str(canary))).replace('WRITABLE', repr(str(writable))).replace('PORT', str(port))
    command = python_nodes._sandbox_command() + ['-c', probe]
    process = subprocess.Popen(command, cwd=tmp_path, env={'PATH': '/usr/bin:/bin', 'LANG': 'C'},
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, close_fds=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=20)
    finally:
        listener.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    assert process.returncode == 0, stderr[:1500]
    assert json.loads(stdout) == {'read': 'denied', 'write': 'denied', 'network': 'denied',
                                  'fork': 'denied', 'environment': 'clean'}
    assert canary.read_text() == 'synthetic-value'
    assert not writable.exists()


@pytest.mark.skipif(not bwrap_available(), reason='bwrap unavailable')
def test_linux_runtime_identity_is_stable_and_names_bwrap(monkeypatch):
    monkeypatch.setattr(python_nodes.subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no process')))
    identity = python_nodes.runtime_identity()
    assert identity['available'] is True
    assert identity['sandbox_executable']['path'].endswith('bwrap')
    assert identity['stdlib']['files'] > 0 and identity['native_extensions']['files'] > 0
    assert identity['os_build']['kernel_release']
    assert identity['interpreter_flags'] == ['-I', '-S', '-B']
    assert python_nodes.runtime_identity() == identity
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_python_nodes_linux.py -q`
Expected: FAIL (`_command_layout` missing; `_sandbox_command` raises on Linux).

- [ ] **Step 3: Implement the Linux path in `auto_jev/python_nodes.py`**

Replace `_sandbox_command()` with:

```python
def _darwin_sandbox_command():
    """Return the actual interpreter plus the reviewed dyld bootstrap profile."""
    if not Path('/usr/bin/sandbox-exec').is_file():
        raise PythonSandboxUnavailable('Functional Python requires the macOS native sandbox')
    base = Path(sys.base_prefix).resolve()
    executable = base / 'Resources/Python.app/Contents/MacOS/Python'
    if not executable.is_file():
        executable = Path(sys._base_executable).resolve()
    libraries = [str(base), '/System/Library', '/usr/lib',
                 '/System/Volumes/Preboot/Cryptexes/OS/System/Library',
                 '/System/Cryptexes/OS/System/Library',
                 '/System/Volumes/Preboot/Cryptexes/OS/usr/lib']
    paths = ' '.join('(subpath ' + json.dumps(path) + ')' for path in libraries)
    profile = '\n'.join([
        '(version 1)', '(deny default)',
        '(allow process-exec (literal ' + json.dumps(str(executable)) + '))',
        '(allow sysctl-read)', '(allow file-read-metadata)',
        '(allow file-read* file-map-executable ' + paths + ')',
        '(deny file-read-data (regex ' + json.dumps(r'\.pyc$') + '))',
        '(allow file-read* (literal "/dev/urandom") (literal "/dev/null"))',
        '(allow file-read* file-test-existence (literal "/"))',
    ])
    return ['/usr/bin/sandbox-exec', '-p', profile, str(executable), '-I', '-S', '-B']


def _linux_sandbox_command():
    """bwrap: empty root, read-only /usr and the interpreter prefix, no network, no host env.

    bwrap cannot deny reads of pre-existing .pyc files by pattern; -B prevents writes and
    runtime_identity() binds the .py sources, so provenance covers what is compiled from.
    """
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise PythonSandboxUnavailable('Functional Python requires bwrap (bubblewrap) on Linux')
    base = Path(sys.base_prefix).resolve()
    executable = Path(sys._base_executable).resolve()
    args = ['--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
            '--setenv', 'PATH', '/usr/bin:/bin', '--setenv', 'LANG', 'C',
            '--ro-bind', '/usr', '/usr',
            '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64', '--symlink', 'usr/bin', '/bin']
    if Path('/etc/ld.so.cache').is_file():
        args += ['--ro-bind', '/etc/ld.so.cache', '/etc/ld.so.cache']
    if not str(base).startswith('/usr/'):
        args += ['--ro-bind', str(base), str(base)]
    if not str(executable).startswith('/usr/') and not str(executable).startswith(str(base)):
        args += ['--ro-bind', str(executable), str(executable)]
    args += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--chdir', '/tmp']
    return [str(Path(bwrap).resolve())] + args + ['--', str(executable), '-I', '-S', '-B']


def _sandbox_command():
    if sys.platform == 'darwin':
        return _darwin_sandbox_command()
    if sys.platform.startswith('linux'):
        return _linux_sandbox_command()
    raise PythonSandboxUnavailable('Functional Python requires a supported native sandbox (macOS or Linux)')


def _command_layout(command):
    """Locate the interpreter, sandbox binary, profile text and flags inside a sandbox command."""
    if command and command[0].endswith('sandbox-exec'):
        return {'executable': command[3], 'sandbox_executable': command[0], 'profile': command[2], 'flags': list(command[4:])}
    separator = command.index('--')
    return {'executable': command[separator + 1], 'sandbox_executable': command[0],
            'profile': json.dumps(command[1:separator], separators=(',', ':')), 'flags': list(command[separator + 2:])}
```

Add `import shutil` to the module imports.

In `runtime_identity()`: replace `command[3]`, `command[0]`, `command[2]`, `command[4:]` with `layout = _command_layout(command)` and `layout['executable']`, `layout['sandbox_executable']`, `layout['profile']`, `layout['flags']`. Replace the darwin-only `framework`/`system_version`/`os_build` lines with:

```python
        extra = {}
        if sys.platform == 'darwin':
            system_version = Path('/System/Library/CoreServices/SystemVersion.plist')
            system_data = plistlib.loads(system_version.read_bytes())
            extra['framework'] = file_identity(base / 'Python')
            extra['system_version'] = file_identity(system_version)
            extra['os_build'] = {key: system_data.get(key) for key in ('ProductName', 'ProductVersion', 'ProductBuildVersion')}
        else:
            release = Path('/etc/os-release')
            fields = {}
            if release.is_file():
                for line in release.read_text().splitlines():
                    if '=' in line:
                        key, value = line.split('=', 1)
                        fields[key.strip()] = value.strip().strip('"')
                extra['os_release'] = file_identity(release)
            extra['os_build'] = {'name': fields.get('NAME'), 'version_id': fields.get('VERSION_ID'),
                                 'pretty_name': fields.get('PRETTY_NAME'), 'kernel_release': os.uname().release}
```

and merge `extra` into `identity.update(...)` (keep every other field: `executable`, `sandbox_executable`, `profile_sha256`, `interpreter_flags`, `stdlib`, `native_extensions`, `stdlib_archive`, `excluded`). Keep the `except (OSError, ValueError, plistlib.InvalidFileException)` clause. The `framework` key must remain darwin-only (the darwin fixture test asserts it; the linux test does not).

In `_BOOTSTRAP`, after the `RLIMIT_DATA` try/except add:

```python
if sys.platform.startswith('linux'):
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT_MEMORY, LIMIT_MEMORY))
```

Rename the existing `_check_memory` to `_check_memory_darwin` and add:

```python
def _check_memory_linux(process):
    """RLIMIT_AS is enforced by the kernel inside the bwrap worker; no host-side RSS polling."""
    return None


_check_memory = _check_memory_darwin if sys.platform == 'darwin' else _check_memory_linux
```

In `evaluate_python`, change the startup-failure detection line to:

```python
            lowered = diagnostic.lower()
            if not output and (b'sandbox' in lowered or b'bwrap' in lowered or process.returncode == 1):
```

- [ ] **Step 4: Mark darwin-shaped upstream tests**

In `tests/test_python_nodes.py`, add `import sys` (present) and decorate with `@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS sandbox-exec layout')`: `test_rusage_struct_layout` (the `_Rusage` size test), `test_memory_reader_failure_is_closed_for_a_live_worker`, `test_memory_threshold_is_a_candidate_error`, `test_launch_contract_discards_host_environment_and_fds`, `test_actual_runtime_os_profile_denies_files_network_and_fork`. In the two memory tests, replace `python_nodes._check_memory(` with `python_nodes._check_memory_darwin(`.

In `tests/test_python_runtime_identity.py`, add at module top after imports:

```python
darwin_only = pytest.mark.skipif(sys.platform != 'darwin', reason='fixture models a macOS framework install')
```

and apply `@darwin_only` to every test that takes the `installation` fixture. In `test_installed_runtime_can_be_fingerprinted_without_starting_workers`, replace `assert identity['os_build']['ProductBuildVersion']` with `assert any(identity['os_build'].values())`.

- [ ] **Step 5: Run the sandbox test files**

Run: `uv run pytest tests/test_python_nodes.py tests/test_python_nodes_linux.py tests/test_python_runtime_identity.py tests/test_pipeline_v3.py tests/test_task_contract.py -q`
Expected: PASS (darwin-only tests show as skipped).

- [ ] **Step 6: Commit**

```bash
git add auto_jev/python_nodes.py tests/test_python_nodes.py tests/test_python_nodes_linux.py tests/test_python_runtime_identity.py
git commit -m "feat: run functional Python nodes under bwrap on Linux"
```

---

### Task 2: JSON-schema subset validator

**Files:**
- Create: `auto_jev/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `validate_json_schema(value, schema) -> value` (raises `ValueError` on mismatch or unsupported keyword), `minimal_instance(schema) -> value`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from auto_jev.schema import minimal_instance, validate_json_schema

PERSON = {'type': 'object', 'required': ['name', 'age'], 'additionalProperties': False,
          'properties': {'name': {'type': 'string', 'minLength': 1},
                         'age': {'type': 'integer', 'minimum': 0, 'maximum': 150},
                         'tags': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 2},
                         'mood': {'type': 'string', 'enum': ['good', 'bad']}}}


def test_valid_object_passes_and_is_returned():
    value = {'name': 'a', 'age': 3, 'tags': ['x'], 'mood': 'good'}
    assert validate_json_schema(value, PERSON) is value


@pytest.mark.parametrize('bad', [
    {'name': 'a'}, {'name': '', 'age': 3}, {'name': 'a', 'age': -1}, {'name': 'a', 'age': 3.5},
    {'name': 'a', 'age': 3, 'tags': ['x', 'y', 'z']}, {'name': 'a', 'age': 3, 'mood': 'meh'},
    {'name': 'a', 'age': 3, 'extra': 1}, 'not an object', {'name': 'a', 'age': True},
])
def test_invalid_values_raise(bad):
    with pytest.raises(ValueError):
        validate_json_schema(bad, PERSON)


def test_type_list_and_null():
    assert validate_json_schema(None, {'type': ['string', 'null']}) is None
    with pytest.raises(ValueError):
        validate_json_schema(1, {'type': ['string', 'null']})


def test_unsupported_keyword_is_rejected_loudly():
    with pytest.raises(ValueError, match='unsupported'):
        validate_json_schema({}, {'type': 'object', 'patternProperties': {}})


def test_minimal_instance_satisfies_schema():
    instance = minimal_instance(PERSON)
    assert validate_json_schema(instance, PERSON) == instance
    assert instance == {'name': 'mock', 'age': 0}
    assert minimal_instance({'type': 'array', 'items': {'type': 'string'}, 'minItems': 3}) == ['mock 1', 'mock 2', 'mock 3']
    assert minimal_instance({'type': 'string', 'enum': ['x', 'y']}) == 'x'
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_schema.py -q` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `auto_jev/schema.py`**

```python
"""A small, explicit JSON Schema subset for LLM node outputs. Unknown keywords fail loudly."""
import math

SUPPORTED = {'type', 'properties', 'required', 'additionalProperties', 'items', 'enum',
             'minimum', 'maximum', 'minLength', 'maxLength', 'minItems', 'maxItems', 'description', 'title'}
TYPES = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'null': type(None)}


def _type_ok(value, name):
    if name == 'number':
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if name == 'integer':
        return isinstance(value, int) and not isinstance(value, bool)
    if name not in TYPES:
        raise ValueError(f'unsupported schema type {name!r}')
    return isinstance(value, TYPES[name])


def validate_json_schema(value, schema, path='$'):
    if not isinstance(schema, dict):
        raise ValueError(f'{path}: schema must be an object')
    unknown = set(schema) - SUPPORTED
    if unknown:
        raise ValueError(f'{path}: unsupported schema keywords {sorted(unknown)}')
    if 'type' in schema:
        names = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        if not any(_type_ok(value, name) for name in names):
            raise ValueError(f'{path}: expected type {schema["type"]}, got {type(value).__name__}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: {value!r} not in enum')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 'minimum' in schema and value < schema['minimum']:
            raise ValueError(f'{path}: {value} < minimum {schema["minimum"]}')
        if 'maximum' in schema and value > schema['maximum']:
            raise ValueError(f'{path}: {value} > maximum {schema["maximum"]}')
    if isinstance(value, str):
        if 'minLength' in schema and len(value) < schema['minLength']:
            raise ValueError(f'{path}: shorter than minLength {schema["minLength"]}')
        if 'maxLength' in schema and len(value) > schema['maxLength']:
            raise ValueError(f'{path}: longer than maxLength {schema["maxLength"]}')
    if isinstance(value, list):
        if 'minItems' in schema and len(value) < schema['minItems']:
            raise ValueError(f'{path}: fewer than minItems {schema["minItems"]}')
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            raise ValueError(f'{path}: more than maxItems {schema["maxItems"]}')
        if 'items' in schema:
            for index, item in enumerate(value):
                validate_json_schema(item, schema['items'], f'{path}[{index}]')
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                raise ValueError(f'{path}: missing required {key!r}')
        for key, child in value.items():
            if key in properties:
                validate_json_schema(child, properties[key], f'{path}.{key}')
            elif schema.get('additionalProperties', True) is False:
                raise ValueError(f'{path}: unexpected property {key!r}')
    return value


def minimal_instance(schema, label='mock'):
    """Smallest value that satisfies the schema; used by the mock LLM."""
    if 'enum' in schema:
        return schema['enum'][0]
    names = schema.get('type', 'object')
    name = names[0] if isinstance(names, list) else names
    if name == 'object':
        return {key: minimal_instance(schema.get('properties', {}).get(key, {'type': 'string'}), label)
                for key in schema.get('required', [])}
    if name == 'array':
        count = schema.get('minItems', 0)
        return [minimal_instance(schema.get('items', {'type': 'string'}), f'{label} {index + 1}') for index in range(count)]
    if name == 'string':
        return label
    if name in ('number', 'integer'):
        return schema.get('minimum', 0)
    if name == 'boolean':
        return False
    if name == 'null':
        return None
    raise ValueError(f'unsupported schema type {name!r}')
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_schema.py -q` — Expected: PASS.

- [ ] **Step 5: Commit** — `git add auto_jev/schema.py tests/test_schema.py && git commit -m "feat: add JSON-schema subset validator for LLM outputs"`

---

### Task 3: `LLMClient` provider

**Files:**
- Create: `auto_jev/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `auto_jev.schema.validate_json_schema`, `minimal_instance`; `auto_jev.storage.atomic_write_json`.
- Produces: `LLMClient(kind='mock', model='mock-llm', api_key_env=None, cache_dir=None, cache_namespace=None, cache_only=False, timeout=60, max_tokens=1024, mock_responder=None)`; `LLMClient.complete(prompt, output_schema=None, model=None, max_tokens=None) -> dict(text, value, model, usage, elapsed_ms, cache_hit, request_hash, cost_usd, client_metadata)`; `LLMClient.stats`, `LLMClient.metadata`, `LLMClient.mock`, `LLMClient.transport`; `LLMProviderError`; `extract_json(text) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
import json

import pytest

from auto_jev.llm import LLMClient, LLMProviderError, extract_json

SCHEMA = {'type': 'object', 'required': ['label'], 'properties': {'label': {'type': 'string', 'enum': ['a', 'b']}}}


def test_mock_builds_minimal_schema_instance_and_counts():
    client = LLMClient(kind='mock')
    result = client.complete('choose', output_schema=SCHEMA)
    assert result['value'] == {'label': 'a'}
    assert json.loads(result['text']) == {'label': 'a'}
    assert result['cache_hit'] is False and result['cost_usd'] is None
    assert client.stats['calls'] == 1 and client.stats['unpriced_calls'] == 0
    assert client.metadata['transport'] == 'mock' and client.mock is True


def test_mock_without_schema_returns_text():
    result = LLMClient(kind='mock').complete('first line\nlast line')
    assert result['value'] == result['text'] == 'last line'


def test_mock_responder_output_is_schema_checked():
    client = LLMClient(kind='mock', mock_responder=lambda prompt, schema: '```json\n{"label": "zzz"}\n```')
    with pytest.raises(LLMProviderError, match='enum'):
        client.complete('x', output_schema=SCHEMA)


def test_fenced_and_surrounded_json_is_extracted():
    assert extract_json('Sure:\n```json\n{"a": 1}\n```\nthanks') == '{"a": 1}'
    assert extract_json('prefix {"a": [1, 2]} suffix') == '{"a": [1, 2]}'


def test_cache_roundtrip_and_cache_only(tmp_path):
    responses = iter(['{"label": "b"}', '{"label": "a"}'])
    client = LLMClient(kind='mock', cache_dir=tmp_path, cache_namespace='ns', mock_responder=lambda p, s: next(responses))
    first = client.complete('same', output_schema=SCHEMA)
    second = client.complete('same', output_schema=SCHEMA)
    assert first['value'] == second['value'] == {'label': 'b'}
    assert second['cache_hit'] is True and client.stats['cache_hits'] == 1
    replay = LLMClient(kind='mock', cache_dir=tmp_path, cache_namespace='ns', cache_only=True)
    assert replay.complete('same', output_schema=SCHEMA)['value'] == {'label': 'b'}
    with pytest.raises(LLMProviderError, match='cache-only'):
        replay.complete('different', output_schema=SCHEMA)


def test_cache_integrity_failure_is_detected(tmp_path):
    client = LLMClient(kind='mock', cache_dir=tmp_path, cache_namespace='ns')
    client.complete('p')
    path = next((tmp_path / 'ns').glob('*.json'))
    tampered = json.loads(path.read_text()); tampered['text'] = 'edited'; path.write_text(json.dumps(tampered))
    with pytest.raises(LLMProviderError, match='integrity'):
        client.complete('p')


def test_real_kinds_require_key_and_never_call_without_it(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    with pytest.raises(LLMProviderError, match='ANTHROPIC_API_KEY'):
        LLMClient(kind='anthropic', model='claude-sonnet-5').complete('x')
    with pytest.raises(ValueError):
        LLMClient(kind='unknown')
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_llm.py -q` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `auto_jev/llm.py`**

```python
"""LLM node transport. Same shape and cache discipline as JevClient; no tools, text in, JSON or text out."""
import copy
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

from .schema import minimal_instance, validate_json_schema
from .storage import atomic_write_json

KINDS = ('mock', 'anthropic', 'openai')
DEFAULT_KEY_ENV = {'anthropic': 'ANTHROPIC_API_KEY', 'openai': 'OPENAI_DIRECT_API_KEY'}
_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.S)


class LLMProviderError(RuntimeError):
    pass


def extract_json(text):
    """Return the JSON document inside fenced or surrounding prose; raise if none is found."""
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith(('{', '[')):
        return stripped
    start = min((index for index in (stripped.find('{'), stripped.find('[')) if index >= 0), default=-1)
    end = max(stripped.rfind('}'), stripped.rfind(']'))
    if start < 0 or end < start:
        raise LLMProviderError('LLM response contains no JSON document')
    return stripped[start:end + 1]


class LLMClient:
    def __init__(self, kind='mock', model='mock-llm', api_key_env=None, cache_dir=None, cache_namespace=None,
                 cache_only=False, timeout=60, max_tokens=1024, mock_responder=None, base_url=None):
        if kind not in KINDS:
            raise ValueError('unknown LLM provider kind')
        self.kind = kind; self.model = model; self.timeout = timeout; self.max_tokens = max_tokens
        self.mock = kind == 'mock'; self.transport = kind
        self.base_url = base_url
        self.key_env = api_key_env or DEFAULT_KEY_ENV.get(kind)
        self.mock_responder = mock_responder
        self.cache_only = cache_only
        self.cache_namespace = cache_namespace or uuid.uuid4().hex
        if not self.cache_namespace.replace('-', '').isalnum():
            raise ValueError('Invalid cache namespace')
        self.cache_dir = Path(cache_dir) / self.cache_namespace if cache_dir else None
        self.stats = {'calls': 0, 'cache_hits': 0, 'input_tokens': 0, 'output_tokens': 0,
                      'elapsed_ms': 0.0, 'cost_usd': 0.0, 'unpriced_calls': 0}
        self.metadata = {'transport': kind, 'requested_model': model, 'model_version_pinned': False,
                         'cache_namespace': self.cache_namespace, 'cache_dir': str(self.cache_dir) if self.cache_dir else None,
                         'api_key_env': self.key_env}

    def complete(self, prompt, output_schema=None, model=None, max_tokens=None):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('prompt must be a nonempty string')
        start = time.perf_counter(); model = model or self.model; max_tokens = max_tokens or self.max_tokens
        payload = {'protocol': 'auto_jev-llm-v1', 'kind': self.kind, 'model': model, 'prompt': prompt,
                   'output_schema': output_schema, 'max_tokens': max_tokens}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        path = self.cache_dir / f'{key}.json' if self.cache_dir else None
        if path and path.exists():
            try:
                result = json.loads(path.read_text())
                digest = result.pop('response_digest', None)
                actual = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
                if digest != actual or result.get('request_hash') != key:
                    raise LLMProviderError('Cache response integrity check failed')
            except (ValueError, TypeError, AttributeError):
                raise LLMProviderError('Cache response integrity check failed') from None
            result['cache_hit'] = True; result['elapsed_ms'] = (time.perf_counter() - start) * 1000
            self.stats['cache_hits'] += 1
            return copy.deepcopy(result)
        if self.cache_only:
            raise LLMProviderError('No recorded response for this input; cache-only replay cannot call the API')
        text, actual_model, usage = self._call(prompt, output_schema, model, max_tokens)
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError('Empty LLM response')
        value = text.strip()
        if output_schema is not None:
            try:
                value = json.loads(extract_json(text))
                validate_json_schema(value, output_schema)
            except ValueError as exc:
                raise LLMProviderError(f'LLM output failed schema validation: {exc}') from None
        result = {'text': text, 'value': value, 'model': actual_model, 'usage': usage, 'cache_hit': False,
                  'request_hash': key, 'elapsed_ms': (time.perf_counter() - start) * 1000, 'cost_usd': None,
                  'client_metadata': {**copy.deepcopy(self.metadata), 'requested_model': model, 'actual_model': actual_model}}
        self.stats['calls'] += 1; self.stats['elapsed_ms'] += result['elapsed_ms']
        self.stats['input_tokens'] += usage.get('input_tokens', 0); self.stats['output_tokens'] += usage.get('output_tokens', 0)
        if not self.mock:
            self.stats['unpriced_calls'] += 1
        if path:
            cached = copy.deepcopy(result)
            cached['response_digest'] = hashlib.sha256(json.dumps(cached, sort_keys=True, allow_nan=False).encode()).hexdigest()
            atomic_write_json(path, cached)
        return copy.deepcopy(result)

    def _call(self, prompt, output_schema, model, max_tokens):
        if self.mock:
            if self.mock_responder is not None:
                return self.mock_responder(prompt, output_schema), 'mock-llm', {}
            if output_schema is not None:
                return json.dumps(minimal_instance(output_schema)), 'mock-llm', {}
            return prompt.strip().splitlines()[-1], 'mock-llm', {}
        key = os.getenv(self.key_env or '')
        if not key:
            raise LLMProviderError(f'Missing {self.key_env}')
        instruction = prompt if output_schema is None else (
            prompt + '\n\nRespond with a single JSON document matching this JSON Schema and nothing else:\n'
            + json.dumps(output_schema, sort_keys=True))
        try:
            if self.kind == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=key, timeout=self.timeout)
                with client:
                    response = client.messages.create(model=model, max_tokens=max_tokens,
                                                      messages=[{'role': 'user', 'content': instruction}])
                text = ''.join(block.text for block in response.content if block.type == 'text')
                usage = {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens}
                return text, response.model, usage
            import openai
            client = openai.OpenAI(api_key=key, max_retries=0, base_url=self.base_url, timeout=self.timeout)
            with client:
                response = client.chat.completions.create(model=model, max_completion_tokens=max_tokens,
                                                          messages=[{'role': 'user', 'content': instruction}])
            text = response.choices[0].message.content or ''
            usage = ({'input_tokens': response.usage.prompt_tokens, 'output_tokens': response.usage.completion_tokens}
                     if response.usage else {})
            return text, response.model, usage
        except Exception as exc:
            raise LLMProviderError(f'LLM request failed: {type(exc).__name__}') from None
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_llm.py -q` — Expected: PASS.

- [ ] **Step 5: Commit** — `git add auto_jev/llm.py tests/test_llm.py && git commit -m "feat: add LLMClient with schema-checked outputs and response cache"`

---

### Task 4: Spec version 4 with `llm` nodes

**Files:**
- Modify: `auto_jev/spec.py` (`validate_spec`, `pipeline_schema`), `auto_jev/flow.py` (`_node_expressions`), `auto_jev/runtime.py` (`PipelineRuntime.__init__`, `_base_record`, `_execute`), `auto_jev/frozen.py` (`source_hash` file list)
- Test: `tests/test_pipeline_v4.py`

**Interfaces:**
- Consumes: `LLMClient.complete`.
- Produces: `PipelineRuntime(spec, jev, llm=None, *, max_workers=8)`; spec v4 llm node `{id, kind:'llm', prompt_expression, output_schema?, max_tokens?, depends_on?}` and top-level `llm_model`; node record fields `prompt`, `output_schema`, `response`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from auto_jev.flow import compile_flow
from auto_jev.llm import LLMClient
from auto_jev.runtime import PipelineExecutionError, PipelineRuntime
from auto_jev.spec import pipeline_schema, validate_spec
from tests.support import ScriptedJev

SCHEMA = {'type': 'object', 'required': ['answer'], 'properties': {'answer': {'type': 'string', 'enum': ['yes', 'no']}}}


def v4(**overrides):
    spec = {'version': 4, 'name': 'llm flow', 'jev_model': 'typesafe-ai/jev', 'llm_model': 'mock-llm',
            'nodes': [
                {'id': 'prompt', 'kind': 'expression', 'expression': "'Question: ' + obs['q']"},
                {'id': 'draft', 'kind': 'llm', 'prompt_expression': "nodes['prompt']", 'output_schema': SCHEMA},
                {'id': 'check', 'kind': 'jev', 'state': "{'draft': nodes['draft']}",
                 'questions': {'ok': {'type': 'noul', 'instructions': 'Is the draft acceptable?'}}},
            ],
            'output': "{'answer': nodes['draft']['answer'], 'ok': nodes['check']['ok']['noul']}"}
    spec.update(overrides)
    return spec


def test_v4_validates_and_infers_llm_dependencies():
    spec = validate_spec(v4())
    assert compile_flow(spec) == {'prompt': [], 'draft': ['prompt'], 'check': ['draft']}
    schema = pipeline_schema(version=4)
    assert schema['properties']['version'] == {'const': 4}
    assert any(kind['properties'].get('kind') == {'const': 'llm'} for kind in schema['properties']['nodes']['items']['oneOf'])


@pytest.mark.parametrize('mutate', [
    lambda s: s.pop('llm_model'),
    lambda s: s['nodes'][1].pop('prompt_expression'),
    lambda s: s['nodes'][1].update(extra=1),
    lambda s: s['nodes'][1].update(output_schema={'type': 'object', 'patternProperties': {}}),
    lambda s: s.update(version=3),
])
def test_invalid_v4_specs_are_rejected(mutate):
    spec = v4(); mutate(spec)
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_v4_runs_llm_then_jev_and_records_prompt_and_response():
    llm = LLMClient(kind='mock', mock_responder=lambda prompt, schema: '{"answer": "yes"}')
    runtime = PipelineRuntime(v4(), ScriptedJev({'ok': 0.9}), llm)
    result = runtime.run({'q': 'go?'})
    assert result['output'] == {'answer': 'yes', 'ok': 0.9}
    record = result['trace'][1]
    assert record['kind'] == 'llm' and record['prompt'] == 'Question: go?'
    assert record['output_schema'] == SCHEMA and record['response']['value'] == {'answer': 'yes'}
    assert llm.stats['calls'] == 1


def test_llm_node_schema_failure_is_a_node_error():
    llm = LLMClient(kind='mock', mock_responder=lambda prompt, schema: '{"answer": "maybe"}')
    runtime = PipelineRuntime(v4(), ScriptedJev(), llm)
    with pytest.raises(PipelineExecutionError) as caught:
        runtime.run({'q': 'go?'})
    trace = caught.value.partial_result['trace']
    assert trace[1]['status'] == 'error' and 'schema' in trace[1]['error']
    assert trace[2]['status'] == 'blocked'


def test_llm_nodes_require_a_client():
    with pytest.raises(ValueError, match='llm'):
        PipelineRuntime(v4(), ScriptedJev())


def test_v3_specs_are_untouched_by_v4_fields():
    with pytest.raises(ValueError):
        validate_spec({**v4(), 'version': 3})
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_pipeline_v4.py -q` — Expected: FAIL (`version must be 1, 2 or 3`).

- [ ] **Step 3: Implement in `auto_jev/spec.py`**

In `validate_spec`:
- Version check: `spec['version'] not in (1,2,3,4)` → message `'pipeline version must be 1, 2, 3 or 4'`.
- `allowed` set: add `'llm_model'`.
- After the name/jev_model checks add:
  ```python
  if 'llm_model' in spec and (not isinstance(spec['llm_model'],str) or not spec['llm_model']):raise ValueError('llm_model must be a nonempty string')
  ```
- Replace every `spec['version']==3` with `spec['version']>=3` (lines using `questions_expression`, strict questions, python nodes).
- Add a new branch before `else:raise ValueError('unknown node kind')`:
  ```python
  elif kind=='llm' and spec['version']>=4:
      if set(node)-{'id','kind','prompt_expression','output_schema','max_tokens','depends_on'}:raise ValueError('unknown LLM node fields')
      if not isinstance(node.get('prompt_expression'),str):raise ValueError('LLM nodes require prompt_expression')
      expression(node['prompt_expression'])
      if 'output_schema' in node:
          from .schema import validate_json_schema, minimal_instance
          if not isinstance(node['output_schema'],dict):raise ValueError('output_schema must be an object')
          validate_json_schema(minimal_instance(node['output_schema']),node['output_schema'])
      if 'max_tokens' in node and (isinstance(node['max_tokens'],bool) or not isinstance(node['max_tokens'],int) or node['max_tokens']<1):raise ValueError('max_tokens must be a positive integer')
  ```
- After the node loop: `if spec['version']>=4 and any(n.get('kind')=='llm' for n in nodes) and 'llm_model' not in spec:raise ValueError('llm nodes require llm_model')`.

In `pipeline_schema`: accept `version in (1,2,3,4)`; keep the `if version==3:` block but change it to `if version>=3:`; then add:

```python
    if version==4:
        schema['properties']['version']={'const':4}
        schema['properties']['llm_model']={'type':'string','description':'Fixed LLM model for llm nodes; required when any llm node exists. Cannot be changed by evolution.'}
        schema['properties']['nodes']['items']['oneOf'].append({'type':'object','required':['id','kind','prompt_expression'],'additionalProperties':False,
            'properties':{'id':{'type':'string'},'kind':{'const':'llm'},'depends_on':{'type':'array','items':{'type':'string'}},
                          'prompt_expression':{'type':'string','description':'Restricted expression evaluating to the prompt string. Dependencies are inferred from nodes[...] references.'},
                          'output_schema':{'type':'object','description':'Optional JSON Schema subset (type, properties, required, additionalProperties, items, enum, minimum, maximum, minLength, maxLength, minItems, maxItems). With a schema the node output is the parsed JSON; without it, the raw text.'},
                          'max_tokens':{'type':'integer','minimum':1}}})
        schema['description']+=(' Version 4 adds llm nodes: a general language model call with a prompt built by expression and an optional output schema. Use llm nodes for open-ended generation (drafts, candidates, structured extraction) and Jev nodes for bounded judgments (choose, score, yes/no). A generate-then-select structure (llm proposes candidates, python turns them into choice criteria, jev selects) is allowed. Node kinds may be swapped when the task allows; the llm_model and jev_model are fixed. Stay within the stated latency and cost budget.')
```

- [ ] **Step 4: Implement in `auto_jev/flow.py`**

In `_node_expressions`, add before the `python` branch:

```python
    if kind == 'llm':
        return [node.get('prompt_expression')]
```

And in `compile_flow`, after the python-node version check add:

```python
        if node.get('kind') == 'llm' and version < 4:
            raise ValueError('LLM nodes require pipeline version 4')
```

- [ ] **Step 5: Implement in `auto_jev/runtime.py`**

`__init__` signature → `def __init__(self, spec, jev, llm=None, *, max_workers=8):`; after `self.jev = jev` add:

```python
        self.llm = llm
        if llm is None and any(node.get('kind') == 'llm' for node in self.spec['nodes']):
            raise ValueError('This pipeline has llm nodes and needs an llm client')
```

`_base_record`: add a branch before the jev `else`:

```python
        elif node['kind'] == 'llm':
            record['prompt_expression'] = node['prompt_expression']
            record['output_schema'] = copy.deepcopy(node.get('output_schema'))
            record['prompt'] = None
            record['response'] = None
```

`_execute`: add before the jev `else`:

```python
            elif node['kind'] == 'llm':
                prompt = evaluate_expression(node['prompt_expression'], context)
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError('prompt_expression must evaluate to a nonempty string')
                record['prompt'] = prompt
                response = copy.deepcopy(self.llm.complete(prompt, output_schema=node.get('output_schema'),
                                                           model=self.spec.get('llm_model'), max_tokens=node.get('max_tokens')))
                record['response'] = response
                result = response['value']
```

- [ ] **Step 6: Bind the new modules into the runtime source hash**

In `auto_jev/frozen.py` `source_hash()`, extend the `files` tuple with `"schema.py", "llm.py"`.

- [ ] **Step 7: Run tests** — `uv run pytest tests/test_pipeline_v4.py tests/test_pipeline_v3.py tests/test_flow.py tests/test_engine.py -q` — Expected: PASS.

- [ ] **Step 8: Commit** — `git add auto_jev/spec.py auto_jev/flow.py auto_jev/runtime.py auto_jev/frozen.py tests/test_pipeline_v4.py && git commit -m "feat: pipeline version 4 with schema-checked llm nodes"`

---

### Task 5: Budgets and `llm` plumbing through evolution and freezing

**Files:**
- Create: `auto_jev/budget.py`
- Modify: `auto_jev/evolution.py` (`run_evolution` signature/config/evaluate/propose), `auto_jev/frozen.py` (`evaluate_frozen`)
- Test: `tests/test_budget.py`, `tests/test_evolution_llm.py`

**Interfaces:**
- Produces: `budget.episode_usage(result) -> dict(latency_ms_p95, latency_ms_median, decisions, cost_usd, unpriced_calls)`, `budget.check_budget(usage, budget) -> list[str]`, `budget.validate_budget(budget) -> dict`; `run_evolution(..., budget=None, llm=None)`; `evaluate_frozen(artifact, episodes, jev, *, evaluator=None, llm=None)`; `evolution.call_evaluator(task, spec, episode, jev, llm, **kwargs)`.

- [ ] **Step 1: Write the failing budget tests (`tests/test_budget.py`)**

```python
import pytest

from auto_jev.budget import check_budget, episode_usage, validate_budget


def result_with(latencies, node_costs=()):
    traces = [{'elapsed_ms': ms, 'trace': [{'kind': 'llm', 'response': {'cost_usd': cost}} for cost in node_costs]} for ms in latencies]
    return {'traces': traces}


def test_usage_percentiles_and_costs():
    usage = episode_usage(result_with([100, 200, 300, 400, 1000], node_costs=[0.01, None]))
    assert usage['decisions'] == 5 and usage['latency_ms_p95'] == 1000 and usage['latency_ms_median'] == 300
    assert usage['cost_usd'] == pytest.approx(0.05) and usage['unpriced_calls'] == 5


def test_usage_without_traces():
    assert episode_usage({}) == {'latency_ms_p95': None, 'latency_ms_median': None, 'decisions': 0, 'cost_usd': None, 'unpriced_calls': 0}


def test_jev_gateway_cost_is_read():
    trace = [{'kind': 'jev', 'response': {'gateway_metadata': {'provider_metadata': {'gateway': {'cost': '0.002'}}}}}]
    assert episode_usage({'traces': [{'elapsed_ms': 5, 'trace': trace}]})['cost_usd'] == pytest.approx(0.002)


def test_check_budget_reports_violations_only_when_measurable():
    usage = episode_usage(result_with([100, 900], node_costs=[0.5]))
    assert check_budget(usage, {'latency_ms_p95': 500, 'cost_usd_per_episode': 0.1}) == [
        'latency_ms_p95 900.0 > 500', 'cost_usd_per_episode 1.0 > 0.1']
    assert check_budget(usage, {'latency_ms_p95': 1000, 'cost_usd_per_episode': 2}) == []
    assert check_budget(episode_usage({}), {'latency_ms_p95': 1}) == []


@pytest.mark.parametrize('bad', [{'latency_ms_p95': -1}, {'unknown': 1}, {'latency_ms_p95': 'fast'}, 'budget'])
def test_validate_budget_rejects(bad):
    with pytest.raises(ValueError):
        validate_budget(bad)
```

- [ ] **Step 2: Implement `auto_jev/budget.py`**

```python
"""Per-episode usage derived from complete traces, and explicit budget checks."""
import math

BUDGET_KEYS = ('latency_ms_p95', 'cost_usd_per_episode')


def validate_budget(budget):
    if not isinstance(budget, dict):
        raise ValueError('budget must be an object')
    if set(budget) - set(BUDGET_KEYS):
        raise ValueError(f'unknown budget keys {sorted(set(budget) - set(BUDGET_KEYS))}')
    for key, value in budget.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'budget {key} must be a non-negative finite number')
    return dict(budget)


def _percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _node_cost(response):
    if not isinstance(response, dict):
        return None
    if response.get('cost_usd') is not None:
        return float(response['cost_usd'])
    gateway = (response.get('gateway_metadata') or {}).get('provider_metadata') or {}
    cost = (gateway.get('gateway') or {}).get('cost')
    return float(cost) if cost is not None else None


def episode_usage(result):
    traces = result.get('traces') if isinstance(result, dict) else None
    traces = traces if isinstance(traces, list) else []
    latencies = [float(t['elapsed_ms']) for t in traces
                 if isinstance(t, dict) and isinstance(t.get('elapsed_ms'), (int, float)) and not isinstance(t['elapsed_ms'], bool)]
    cost, unpriced, priced = 0.0, 0, False
    for decision in traces:
        for node in (decision.get('trace') or []) if isinstance(decision, dict) else []:
            if not isinstance(node, dict) or node.get('kind') not in ('jev', 'llm'):
                continue
            node_cost = _node_cost(node.get('response'))
            if node_cost is None:
                unpriced += 1
            else:
                cost += node_cost; priced = True
    return {'latency_ms_p95': _percentile(latencies, 0.95) if latencies else None,
            'latency_ms_median': _percentile(latencies, 0.5) if latencies else None,
            'decisions': len(traces), 'cost_usd': cost if priced else None, 'unpriced_calls': unpriced}


def check_budget(usage, budget):
    budget = validate_budget(budget)
    violations = []
    if 'latency_ms_p95' in budget and usage.get('latency_ms_p95') is not None and usage['latency_ms_p95'] > budget['latency_ms_p95']:
        violations.append(f"latency_ms_p95 {usage['latency_ms_p95']} > {budget['latency_ms_p95']}")
    if 'cost_usd_per_episode' in budget and usage.get('cost_usd') is not None and usage['cost_usd'] > budget['cost_usd_per_episode']:
        violations.append(f"cost_usd_per_episode {usage['cost_usd']} > {budget['cost_usd_per_episode']}")
    return violations
```

- [ ] **Step 3: Run budget tests** — `uv run pytest tests/test_budget.py -q` — Expected: PASS.

- [ ] **Step 4: Write the failing evolution tests (`tests/test_evolution_llm.py`)**

```python
import json

import pytest

from auto_jev.evolution import FAILURE_SCORE, freeze_run, run_evolution
from auto_jev.frozen import evaluate_frozen
from auto_jev.llm import LLMClient
from auto_jev.runtime import PipelineRuntime
from auto_jev.storage import RunStore
from tests.support import ScriptedJev

SPEC = {'version': 4, 'name': 'echo', 'jev_model': 'typesafe-ai/jev', 'llm_model': 'mock-llm',
        'nodes': [{'id': 'draft', 'kind': 'llm', 'prompt_expression': "obs['text']",
                   'output_schema': {'type': 'object', 'required': ['label'], 'properties': {'label': {'type': 'string'}}}}],
        'output': "nodes['draft']['label']"}


def evaluate_with_llm(spec, episode, jev, llm=None, *, capture_traces=True):
    decision = PipelineRuntime(spec, jev, llm).run({'text': episode['text']})
    return {'episode_id': episode['id'], 'score': float(decision['output'] == episode['label']), 'traces': [decision]}


def evaluate_legacy(spec, episode, jev, capture_traces=True):
    return {'episode_id': episode['id'], 'score': 1.0, 'traces': [{'elapsed_ms': 5, 'trace': []}]}


def proposer(prompt):
    payload = json.loads(prompt)
    assert 'budget' in payload['task'] and payload['task']['budget'] == {'latency_ms_p95': 5000}
    assert 'llm nodes' in payload['instruction']
    return json.dumps(payload['parent'])


def data():
    train = [{'id': f't{i}', 'text': 'x', 'label': 'x'} for i in range(2)]
    return train, [{'id': 'v0', 'text': 'x', 'label': 'x'}]


def test_llm_is_passed_to_evaluators_that_accept_it_and_frozen(tmp_path):
    train, validation = data()
    llm = LLMClient(kind='mock', mock_responder=lambda p, s: '{"label": "x"}')
    store = RunStore(tmp_path)
    summary = run_evolution(train, validation, jev=ScriptedJev(), llm=llm, proposer=proposer, store=store,
                            seed_pipeline=SPEC, evaluator=evaluate_with_llm, task_id='echo', costs={},
                            evolution_rounds=1, reflection_batch_size=1, budget={'latency_ms_p95': 5000})
    assert summary['best_score'] == 1.0
    assert store.get_run(summary['run_id'])['config']['budget'] == {'latency_ms_p95': 5000}
    artifact = freeze_run(store, summary['run_id'])
    assert artifact['budget'] == {'latency_ms_p95': 5000} and artifact['llm_metadata']['transport'] == 'mock'
    outputs = evaluate_frozen(artifact, validation, ScriptedJev(), evaluator=evaluate_with_llm, llm=llm)
    assert outputs[0]['score'] == 1.0
    with pytest.raises(ValueError, match='llm'):
        evaluate_frozen(artifact, validation, ScriptedJev(), evaluator=evaluate_with_llm)


def test_legacy_evaluator_without_llm_param(tmp_path):
    train, validation = data()
    summary = run_evolution(train, validation, jev=ScriptedJev(), llm=LLMClient(kind='mock'), proposer=proposer,
                            store=RunStore(tmp_path), seed_pipeline={**SPEC, 'nodes': [], 'output': "'x'"},
                            evaluator=evaluate_legacy, task_id='legacy', costs={}, evolution_rounds=1,
                            reflection_batch_size=1, budget={'latency_ms_p95': 5000})
    assert summary['best_score'] == 1.0


def test_over_budget_episode_is_invalid_and_recorded(tmp_path):
    train, validation = data()
    store = RunStore(tmp_path)
    with pytest.raises(ValueError, match='validation'):
        run_evolution(train, validation, jev=ScriptedJev(), llm=LLMClient(kind='mock'), proposer=proposer,
                      store=store, seed_pipeline={**SPEC, 'nodes': [], 'output': "'x'"}, evaluator=evaluate_legacy,
                      task_id='slow', costs={}, evolution_rounds=1, reflection_batch_size=1, budget={'latency_ms_p95': 1})
    run_id = store.list_runs()[0]['run_id']
    rows = store.list_evaluations(run_id, split='validation')
    assert rows[0]['status'] == 'over_budget' and rows[0]['score'] == FAILURE_SCORE
    assert rows[0]['usage']['latency_ms_p95'] == 5.0 and 'latency_ms_p95' in rows[0]['error']


def test_budget_not_applied_to_failed_episode(tmp_path):
    def broken(spec, episode, jev, capture_traces=True):
        raise ValueError('candidate produced an illegal action')
    train, validation = data()
    store = RunStore(tmp_path)
    with pytest.raises(ValueError):
        run_evolution(train, validation, jev=ScriptedJev(), proposer=proposer, store=store,
                      seed_pipeline={**SPEC, 'nodes': [], 'output': "'x'"}, evaluator=broken, task_id='broken',
                      costs={}, evolution_rounds=1, reflection_batch_size=1, budget={'latency_ms_p95': 0})
    run_id = store.list_runs()[0]['run_id']
    assert store.list_evaluations(run_id, split='validation')[0]['status'] == 'error'
```

- [ ] **Step 5: Implement in `auto_jev/evolution.py`**

Add imports: `from .budget import check_budget, episode_usage, validate_budget`.

Add a module-level helper after `_digest`:

```python
def call_evaluator(task,spec,episode,jev,llm=None,**kwargs):
    """Pass llm only to evaluators that declare it, so legacy adapters keep working."""
    if 'llm' in inspect.signature(task).parameters:return task(spec,episode,jev,llm=llm,**kwargs)
    return task(spec,episode,jev,**kwargs)
```

`run_evolution` signature: append `,budget=None,llm=None` after `skip_perfect_score=False`. After the `skip_perfect_score` validation add `budget=validate_budget(budget) if budget is not None else None`.

In `config`, add `'budget':budget,'llm_mock':getattr(llm,'mock',None) if llm is not None else None`. In the resume contract loop's key tuple add `'budget'`. In `store.update_run(run_id,status='running',...)` add `llm_metadata=getattr(llm,'metadata',{}) if llm is not None else {}`; do the same in the two later `store.update_run(...)` calls (success and failure).

In `Adapter.evaluate`, change `result=task(spec,ep,jev,capture_traces=True,**costs)` to `result=call_evaluator(task,spec,ep,jev,llm,capture_traces=True,**costs)`. Immediately after `status='ok'` (still inside the `try`), add:

```python
                    usage=episode_usage(result)
                    violations=check_budget(usage,budget) if budget else []
                    if violations:
                        score=FAILURE_SCORE;status='over_budget'
                        result=copy.deepcopy(result);result.update(score=score,error='over budget: '+'; '.join(violations))
```

In the `record={...}` dict add `'usage':episode_usage(result)`.

In `propose_impl`, add to `payload['task']`: `'budget':budget`. Extend `version_instruction` with a v4 branch:

```python
        if pipeline['version']>=4:version_instruction+=('Version 4 also allows llm nodes (open-ended generation with prompt_expression and an optional output_schema). Use Jev nodes for bounded judgments and llm nodes for generation; a generate-then-select structure is allowed. Node kinds may be swapped where the task allows. Never change llm_model. Stay within the stated task budget: latency and cost violations make a candidate invalid. ')
```

Also require `new.get('llm_model')==pipeline.get('llm_model')` next to the `jev_model` check with message `'Cannot change the fixed LLM model'`.

In `freeze_run`, add to `artifact`: `'budget':config.get('budget'),'llm_metadata':run.get('llm_metadata',{})`. Make `progress()` in `_EvolutionRounds` also record `llm=dict(getattr(self.llm,'stats',{}))` and set `rounds.llm=llm` next to `rounds.jev,rounds.proposer=jev,proposer` (add `self.llm=None` in `__init__`).

- [ ] **Step 6: Implement in `auto_jev/frozen.py`**

```python
def evaluate_frozen(artifact, episodes, jev, *, evaluator=None, llm=None):
    if artifact["task_id"] != "crypto_spot" and evaluator is None:
        raise ValueError("This task requires its evaluator")
    task = evaluator or evaluate_episode
    validate_artifact(artifact, jev, evaluator=task)
    needs_llm = any(node.get('kind') == 'llm' for node in artifact['spec'].get('nodes', []))
    if needs_llm and llm is None:
        raise ValueError('This frozen artifact has llm nodes and needs an llm client')
    expected = (artifact.get('llm_metadata') or {}).get('transport')
    if llm is not None and expected and expected != ('mock' if getattr(llm, 'mock', False) else getattr(llm, 'transport', None)):
        raise ValueError('LLM transport differs from the frozen artifact')
    from .evolution import call_evaluator
    return [call_evaluator(task, artifact["spec"], episode, jev, llm, capture_traces=True, **artifact["costs"]) for episode in episodes]
```

(`evolution` imports `frozen`; the local import inside the function avoids a cycle.)

- [ ] **Step 7: Run** — `uv run pytest tests/test_budget.py tests/test_evolution_llm.py tests/test_evolution_rounds.py tests/test_engine.py tests/test_experiment.py -q` — Expected: PASS.

- [ ] **Step 8: Commit** — `git add auto_jev/budget.py auto_jev/evolution.py auto_jev/frozen.py tests/test_budget.py tests/test_evolution_llm.py && git commit -m "feat: budgets and llm client plumbing through evolution and freezing"`

---

### Task 6: Triage — probe, Jev route decision, seed templates

**Files:**
- Create: `auto_jev/triage.py`
- Test: `tests/test_triage.py`

**Interfaces:**
- Consumes: `spec.validate_spec`, `spec.validate_questions`, a Jev client with `.judge(state, questions, model=None)`.
- Produces: `probe(task) -> dict`, `DEFAULT_TRIAGE_QUESTIONS`, `triage(task, jev, *, questions=None, jev_model='typesafe-ai/jev') -> RouteDecision`, `seed_for_route(route, task) -> spec`, `route_of_spec(spec) -> str`, `validate_task(task) -> task`, `ROUTES = ('settled','open_ended','hybrid')`.

- [ ] **Step 1: Write the failing tests**

```python
import copy

import pytest

from auto_jev.llm import LLMClient
from auto_jev.runtime import PipelineRuntime
from auto_jev.spec import validate_spec
from auto_jev.triage import DEFAULT_TRIAGE_QUESTIONS, ROUTES, probe, route_of_spec, seed_for_route, triage, validate_task
from tests.support import ScriptedJev

LABEL_TASK = {'name': 'intents', 'objective': 'Route a message to the right intent.',
              'observation_fields': ['text'],
              'action_space': {'kind': 'labels', 'labels': ['card_lost', 'refund', 'other']},
              'evaluator': {'kind': 'exact_match'}, 'decisions_per_episode': 1, 'latency_budget_ms': 800,
              'examples': [{'observation': {'text': 'I lost my card'}, 'expected': 'card_lost'},
                           {'observation': {'text': 'money back please'}, 'expected': 'refund'}]}

TEXT_TASK = {'name': 'reply', 'objective': 'Write a reply that satisfies the constraints.',
             'observation_fields': ['instruction', 'constraints'],
             'action_space': {'kind': 'open'}, 'evaluator': {'kind': 'code_check'},
             'decisions_per_episode': 1, 'latency_budget_ms': None,
             'output_schema': {'type': 'object', 'required': ['text'], 'properties': {'text': {'type': 'string', 'minLength': 1}}},
             'examples': [{'observation': {'instruction': 'Say hi in under twenty words to a customer who waited.', 'constraints': []},
                           'expected': 'Hello, and thank you for your patience today.'}]}


def test_validate_task_rejects_missing_fields():
    bad = copy.deepcopy(LABEL_TASK); bad.pop('examples')
    with pytest.raises(ValueError, match='examples'):
        validate_task(bad)
    with pytest.raises(ValueError):
        validate_task({**LABEL_TASK, 'action_space': {'kind': 'labels'}})


def test_probe_is_deterministic_code_evidence():
    evidence = probe(LABEL_TASK)
    assert evidence == {'declared_action_kind': 'labels', 'label_count': 3, 'catalog_present_in_examples': False,
                        'exact_validator': True, 'expected_kind': 'label', 'median_expected_tokens': 1,
                        'decisions_per_episode': 1, 'latency_budget_ms': 800, 'example_count': 2}
    text = probe(TEXT_TASK)
    assert text['expected_kind'] == 'short_text' and text['exact_validator'] is True and text['label_count'] is None


def test_triage_sends_evidence_and_maps_confident_answer():
    jev = ScriptedJev({'route': {'settled': 0.9, 'open_ended': 0.05, 'hybrid': 0.05}, 'code_computes_facts': 0.8, 'needs_multistep': 0.1})
    decision = triage(LABEL_TASK, jev)
    assert decision['route'] == 'settled' and decision['confidence'] == 0.9
    assert decision['probabilities'] == {'settled': 0.9, 'open_ended': 0.05, 'hybrid': 0.05}
    assert decision['evidence'] == probe(LABEL_TASK)
    sent = jev.requests[0]
    assert sent['state']['evidence'] == probe(LABEL_TASK) and len(sent['state']['examples']) == 2
    assert set(sent['questions']) == set(DEFAULT_TRIAGE_QUESTIONS)
    assert decision['questions'] == DEFAULT_TRIAGE_QUESTIONS


def test_ambiguous_triage_still_returns_argmax_with_low_confidence():
    jev = ScriptedJev({'route': {'settled': 0.1, 'open_ended': 0.5, 'hybrid': 0.4}})
    decision = triage(TEXT_TASK, jev)
    assert decision['route'] == 'open_ended' and decision['confidence'] == 0.5


def test_triage_rejects_unknown_route():
    class Broken(ScriptedJev):
        def judge(self, state, questions, model=None):
            answer = super().judge(state, questions, model)
            answer['answers']['route']['choice'] = 'teleport'
            return answer
    with pytest.raises(ValueError, match='route'):
        triage(LABEL_TASK, Broken())


@pytest.mark.parametrize('route', ROUTES)
def test_seed_templates_validate_and_report_route(route):
    task = LABEL_TASK if route == 'settled' else TEXT_TASK
    spec = validate_spec(seed_for_route(route, task))
    assert spec['version'] == 4
    assert route_of_spec(spec) == {'settled': 'jev', 'open_ended': 'llm', 'hybrid': 'hybrid'}[route]


def sandbox_ok():
    from auto_jev.python_nodes import PythonSandboxUnavailable, _sandbox_command
    try:
        _sandbox_command(); return True
    except PythonSandboxUnavailable:
        return False


@pytest.mark.skipif(not sandbox_ok(), reason='needs the Python sandbox')
def test_settled_seed_runs_and_returns_a_label():
    spec = seed_for_route('settled', LABEL_TASK)
    jev = ScriptedJev({'action': {'card_lost': 0.1, 'refund': 0.8, 'other': 0.1}})
    result = PipelineRuntime(spec, jev).run({'text': 'money back please', 'labels': LABEL_TASK['action_space']['labels']})
    assert result['output'] == 'refund'
    assert set(jev.requests[0]['questions']['action']['criteria']) == {'card_lost', 'refund', 'other'}


@pytest.mark.skipif(not sandbox_ok(), reason='needs the Python sandbox')
def test_open_ended_seed_runs_with_schema():
    spec = seed_for_route('open_ended', TEXT_TASK)
    llm = LLMClient(kind='mock', mock_responder=lambda p, s: '{"text": "Hello and thanks for waiting."}')
    result = PipelineRuntime(spec, ScriptedJev(), llm).run({'instruction': 'greet', 'constraints': []})
    assert result['output'] == {'text': 'Hello and thanks for waiting.'}
    assert 'greet' in result['trace'][1]['prompt']


@pytest.mark.skipif(not sandbox_ok(), reason='needs the Python sandbox')
def test_hybrid_seed_generates_then_selects():
    spec = seed_for_route('hybrid', TEXT_TASK)
    llm = LLMClient(kind='mock', mock_responder=lambda p, s: '{"candidates": ["short", "a much longer candidate reply", "medium reply"]}')
    jev = ScriptedJev({'best': {'c0': 0.1, 'c1': 0.2, 'c2': 0.7}})
    result = PipelineRuntime(spec, jev, llm).run({'instruction': 'greet', 'constraints': []})
    assert result['output'] == {'text': 'medium reply'}
    criteria = jev.requests[0]['questions']['best']['criteria']
    assert criteria == {'c0': 'short', 'c1': 'a much longer candidate reply', 'c2': 'medium reply'}
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_triage.py -q` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `auto_jev/triage.py`**

```python
"""Route a task before authoring: code probes evidence, Jev judges the route, templates seed each route."""
import copy
import json
import statistics

from .spec import _json, validate_questions, validate_spec

ROUTES = ('settled', 'open_ended', 'hybrid')
ACTION_KINDS = ('labels', 'catalog', 'open', 'declared_unknown')
EVALUATOR_KINDS = ('exact_match', 'environment', 'code_check', 'reward_model', 'judge')
EXACT_EVALUATORS = ('exact_match', 'environment', 'code_check')
SHORT_TEXT_TOKENS = 12
EXAMPLE_LIMIT = 5
EXAMPLE_CHARS = 300

DEFAULT_TRIAGE_QUESTIONS = {
    'route': {
        'type': 'choice',
        'instructions': ('Using the probe evidence and examples, judge which decision form fits this task. '
                         'Evidence fields are computed by code from the task description; examples are representative.'),
        'criteria': {
            'settled': 'Every decision picks from a finite, enumerable set (labels or a per-step legal action list); '
                       'the output can be checked exactly by code; the judgment is choose, score or yes/no.',
            'open_ended': 'The output is free text, code or an unbounded structure that must be generated, '
                          'not selected; no small candidate set exists before the model writes it.',
            'hybrid': 'The output is open, but a few candidates can be generated first and then the best one '
                      'selected by comparison; quality is judged by comparing options.',
        },
    },
    'code_computes_facts': {'type': 'noul', 'instructions': 'Can deterministic code compute most facts needed for each decision, leaving only a bounded judgment to a model?'},
    'needs_multistep': {'type': 'noul', 'instructions': 'Does a single decision require multi-step reasoning or tool use before an answer can be given?'},
}


def validate_task(task):
    _json(task)
    if not isinstance(task, dict):
        raise ValueError('task must be an object')
    for key in ('name', 'objective', 'observation_fields', 'action_space', 'evaluator', 'decisions_per_episode', 'examples'):
        if key not in task:
            raise ValueError(f'task requires {key}')
    if not isinstance(task['examples'], list) or not task['examples']:
        raise ValueError('task examples must be a nonempty list')
    for example in task['examples']:
        if not isinstance(example, dict) or 'observation' not in example or 'expected' not in example:
            raise ValueError('each example needs observation and expected')
    action = task['action_space']
    if not isinstance(action, dict) or action.get('kind') not in ACTION_KINDS:
        raise ValueError('action_space.kind must be one of ' + ', '.join(ACTION_KINDS))
    if action['kind'] == 'labels' and (not isinstance(action.get('labels'), list) or not action['labels']):
        raise ValueError('labels action space needs a nonempty labels list')
    if action['kind'] == 'catalog' and not isinstance(action.get('catalog_field'), str):
        raise ValueError('catalog action space needs catalog_field')
    if not isinstance(task['evaluator'], dict) or task['evaluator'].get('kind') not in EVALUATOR_KINDS:
        raise ValueError('evaluator.kind must be one of ' + ', '.join(EVALUATOR_KINDS))
    if isinstance(task['decisions_per_episode'], bool) or not isinstance(task['decisions_per_episode'], int) or task['decisions_per_episode'] < 1:
        raise ValueError('decisions_per_episode must be a positive integer')
    if 'output_schema' in task and not isinstance(task['output_schema'], dict):
        raise ValueError('output_schema must be an object')
    return copy.deepcopy(task)


def _token_count(value):
    return len(str(value).split()) if isinstance(value, str) else len(json.dumps(value))


def probe(task):
    """Deterministic evidence from the task description; no model calls."""
    task = validate_task(task)
    action = task['action_space']
    labels = action.get('labels') if action['kind'] == 'labels' else None
    field = action.get('catalog_field')
    expected = [example['expected'] for example in task['examples']]
    if labels is not None and all(value in labels for value in expected):
        expected_kind = 'label'
    elif all(isinstance(value, str) for value in expected):
        median = statistics.median(_token_count(value) for value in expected)
        expected_kind = 'short_text' if median <= SHORT_TEXT_TOKENS else 'long_text'
    else:
        expected_kind = 'structured'
    return {
        'declared_action_kind': action['kind'],
        'label_count': len(labels) if labels is not None else None,
        'catalog_present_in_examples': bool(field) and all(
            isinstance(example['observation'].get(field), list) and example['observation'][field] for example in task['examples']),
        'exact_validator': task['evaluator']['kind'] in EXACT_EVALUATORS,
        'expected_kind': expected_kind,
        'median_expected_tokens': statistics.median(_token_count(value) for value in expected),
        'decisions_per_episode': task['decisions_per_episode'],
        'latency_budget_ms': task.get('latency_budget_ms'),
        'example_count': len(task['examples']),
    }


def _clip(value):
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= EXAMPLE_CHARS else text[:EXAMPLE_CHARS] + '…'


def triage(task, jev, *, questions=None, jev_model='typesafe-ai/jev'):
    task = validate_task(task)
    questions = validate_questions(copy.deepcopy(questions or DEFAULT_TRIAGE_QUESTIONS), strict=True)
    if set(questions['route']['criteria']) != set(ROUTES):
        raise ValueError('triage route criteria must be exactly settled, open_ended, hybrid')
    evidence = probe(task)
    state = {'objective': task['objective'], 'evidence': evidence,
             'examples': [{'observation': _clip(example['observation']), 'expected': _clip(example['expected'])}
                          for example in task['examples'][:EXAMPLE_LIMIT]]}
    response = copy.deepcopy(jev.judge(state, copy.deepcopy(questions), model=jev_model))
    answer = response['answers']['route']
    probabilities = {route: float(answer.get('probabilities', {}).get(route, 0.0)) for route in ROUTES}
    if answer.get('choice') not in ROUTES:
        raise ValueError(f"triage returned an unknown route {answer.get('choice')!r}")
    route = answer['choice']
    return {'route': route, 'probabilities': probabilities, 'confidence': probabilities[route],
            'evidence': evidence, 'questions': questions, 'response': response}


def route_of_spec(spec):
    kinds = {node.get('kind') for node in spec.get('nodes', [])}
    if 'jev' in kinds and 'llm' in kinds:
        return 'hybrid'
    if 'jev' in kinds:
        return 'jev'
    if 'llm' in kinds:
        return 'llm'
    return 'code'


_SETTLED_CATALOG = '''def run(obs, nodes, memory):
    options = {}
    if isinstance(obs.get("labels"), list) and obs["labels"]:
        for label in obs["labels"]:
            options[str(label)] = "The correct answer is " + str(label) + "."
    else:
        for item in obs.get(CATALOG_FIELD) or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                options[item["id"]] = str(item)
            elif isinstance(item, str):
                options[item] = item
    if not options:
        options["none"] = "No option was available."
    return {"options": list(options), "questions": {"action": {
        "type": "choice",
        "instructions": OBJECTIVE + " Choose exactly one option ID for the observation.",
        "criteria": options}}}
'''

_SETTLED_DECISION = '''def run(obs, nodes, memory):
    options = nodes["catalog"]["options"]
    chosen = nodes["pick"]["action"].get("choice")
    if chosen in options:
        return chosen
    return options[0]
'''

_OPEN_PROMPT = '''def run(obs, nodes, memory):
    lines = [OBJECTIVE, ""]
    for field in FIELDS:
        lines.append(field + ": " + str(obs.get(field)))
    lines.append("")
    lines.append("Write the best output for this input.")
    return {"prompt": "\\n".join(lines)}
'''

_OPEN_VALIDATE = '''def run(obs, nodes, memory):
    return nodes["draft"]
'''

_HYBRID_PROMPT = '''def run(obs, nodes, memory):
    lines = [OBJECTIVE, ""]
    for field in FIELDS:
        lines.append(field + ": " + str(obs.get(field)))
    lines.append("")
    lines.append("Write 3 different candidate outputs for this input.")
    return {"prompt": "\\n".join(lines)}
'''

_HYBRID_CRITERIA = '''def run(obs, nodes, memory):
    candidates = nodes["propose"]["candidates"]
    criteria = {}
    for index, candidate in enumerate(candidates):
        text = str(candidate).strip()
        criteria["c" + str(index)] = text if text else "(empty candidate)"
    return {"candidates": candidates, "questions": {"best": {
        "type": "choice",
        "instructions": OBJECTIVE + " Choose the candidate that best serves this input.",
        "criteria": criteria}}}
'''

_HYBRID_DECISION = '''def run(obs, nodes, memory):
    candidates = nodes["criteria"]["candidates"]
    chosen = nodes["select"]["best"].get("choice") or "c0"
    index = int(chosen[1:]) if chosen[1:].isdigit() and int(chosen[1:]) < len(candidates) else 0
    return {"text": str(candidates[index])}
'''


def _fill(source, task):
    fields = task['observation_fields']
    return (source.replace('OBJECTIVE', json.dumps(task['objective']))
                  .replace('FIELDS', json.dumps(fields))
                  .replace('CATALOG_FIELD', json.dumps(task['action_space'].get('catalog_field') or 'legal_actions')))


def seed_for_route(route, task):
    task = validate_task(task)
    fields_expr = '{' + ', '.join(f"{json.dumps(f)}: obs[{json.dumps(f)}]" for f in task['observation_fields']) + '}'
    if route == 'settled':
        spec = {'version': 4, 'name': f"{task['name']} · settled seed", 'jev_model': 'typesafe-ai/jev',
                'nodes': [
                    {'id': 'catalog', 'kind': 'python', 'source': _fill(_SETTLED_CATALOG, task), 'depends_on': []},
                    {'id': 'pick', 'kind': 'jev', 'state': fields_expr, 'questions_expression': "nodes['catalog']['questions']"},
                    {'id': 'decision', 'kind': 'python', 'source': _SETTLED_DECISION, 'depends_on': ['catalog', 'pick']}],
                'output': "nodes['decision']"}
    elif route == 'open_ended':
        draft = {'id': 'draft', 'kind': 'llm', 'prompt_expression': "nodes['prompt']['prompt']"}
        if 'output_schema' in task:
            draft['output_schema'] = task['output_schema']
        spec = {'version': 4, 'name': f"{task['name']} · open-ended seed", 'jev_model': 'typesafe-ai/jev', 'llm_model': 'mock-llm',
                'nodes': [
                    {'id': 'prompt', 'kind': 'python', 'source': _fill(_OPEN_PROMPT, task), 'depends_on': []},
                    draft,
                    {'id': 'validate', 'kind': 'python', 'source': _OPEN_VALIDATE, 'depends_on': ['draft']}],
                'output': "nodes['validate']"}
    elif route == 'hybrid':
        spec = {'version': 4, 'name': f"{task['name']} · hybrid seed", 'jev_model': 'typesafe-ai/jev', 'llm_model': 'mock-llm',
                'nodes': [
                    {'id': 'prompt', 'kind': 'python', 'source': _fill(_HYBRID_PROMPT, task), 'depends_on': []},
                    {'id': 'propose', 'kind': 'llm', 'prompt_expression': "nodes['prompt']['prompt']",
                     'output_schema': {'type': 'object', 'required': ['candidates'], 'properties': {
                         'candidates': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'minItems': 2, 'maxItems': 5}}}},
                    {'id': 'criteria', 'kind': 'python', 'source': _fill(_HYBRID_CRITERIA, task), 'depends_on': ['propose']},
                    {'id': 'select', 'kind': 'jev', 'state': fields_expr, 'questions_expression': "nodes['criteria']['questions']"},
                    {'id': 'decision', 'kind': 'python', 'source': _HYBRID_DECISION, 'depends_on': ['criteria', 'select']}],
                'output': "nodes['decision']"}
    else:
        raise ValueError('unknown route')
    return validate_spec(spec)
```

Note for the implementer: the settled seed reads `obs["labels"]` first, so a labels task must put the label list into the observation (the classify adapter in Task 7 does). The `llm_model` in seeds is `'mock-llm'`; `run_routed_evolution` (Task 8) overwrites it with the model name from its `llm` client.

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_triage.py -q` — Expected: PASS (sandbox tests skip if no bwrap).

- [ ] **Step 5: Commit** — `git add auto_jev/triage.py tests/test_triage.py && git commit -m "feat: task probe, Jev triage and per-route seed templates"`

---

### Task 7: Example adapters with fixtures

**Files:**
- Create: `examples/__init__.py`, `examples/classify/__init__.py`, `examples/classify/adapter.py`, `examples/classify/task.json`, `examples/classify/fixture.json`, `examples/constrained_text/__init__.py`, `examples/constrained_text/adapter.py`, `examples/constrained_text/task.json`, `examples/constrained_text/fixture.json`
- Test: `tests/test_example_adapters.py`

**Interfaces:**
- Produces: `examples.classify.adapter.evaluate_episode(spec, episode, jev, llm=None, *, capture_traces=True)`, `examples.classify.adapter.load(split) -> (task, episodes)`; same names in `examples.constrained_text.adapter`; `examples.constrained_text.adapter.score_constraints(text, constraints) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from examples.classify import adapter as classify
from examples.constrained_text import adapter as constrained
from auto_jev.runtime import PipelineExecutionError
from tests.support import ScriptedJev

CONST = {'version': 2, 'name': 'const', 'jev_model': 'typesafe-ai/jev', 'nodes': [], 'output': "'refund'"}


def test_classify_fixture_splits_are_disjoint_and_labelled():
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    assert len(train) == 24 and len(validation) == 16
    assert {e['id'] for e in train}.isdisjoint({e['id'] for e in validation})
    labels = set(task['action_space']['labels'])
    assert all(e['label'] in labels and e['labels'] == task['action_space']['labels'] for e in train + validation)


def test_classify_scores_exact_label_and_hides_label_from_observation():
    _, train = classify.load('train')
    episode = next(e for e in train if e['label'] == 'refund')
    result = classify.evaluate_episode(CONST, episode, ScriptedJev())
    assert result['score'] == 1.0 and result['episode_id'] == episode['id']
    assert 'label' not in result['traces'][0]['input_context']['obs']
    other = next(e for e in train if e['label'] != 'refund')
    assert classify.evaluate_episode(CONST, other, ScriptedJev())['score'] == 0.0


def test_classify_non_label_output_is_a_candidate_error():
    _, train = classify.load('train')
    with pytest.raises(PipelineExecutionError) as caught:
        classify.evaluate_episode({**CONST, 'output': "'nonsense'"}, train[0], ScriptedJev())
    assert not getattr(caught.value.cause, 'task_infrastructure_error', False)


@pytest.mark.parametrize('text,constraints,expected', [
    ('one two three', [{'kind': 'max_words', 'value': 3}], 1.0),
    ('one two three four', [{'kind': 'max_words', 'value': 3}], 0.0),
    ('hello world', [{'kind': 'must_include', 'value': 'world'}, {'kind': 'must_not_include', 'value': 'bye'}], 1.0),
    ('hello', [{'kind': 'min_words', 'value': 2}, {'kind': 'ends_with', 'value': 'o'}], 0.5),
])
def test_constraint_scoring(text, constraints, expected):
    assert constrained.score_constraints(text, constraints) == expected


def test_constrained_evaluator_requires_text_object():
    _, train = constrained.load('train')
    ok = {**CONST, 'output': "{'text': 'thank you for waiting today'}"}
    result = constrained.evaluate_episode(ok, train[0], ScriptedJev())
    assert 0.0 <= result['score'] <= 1.0 and result['output'] == {'text': 'thank you for waiting today'}
    with pytest.raises(PipelineExecutionError):
        constrained.evaluate_episode({**CONST, 'output': "'bare string'"}, train[0], ScriptedJev())


def test_unknown_constraint_kind_is_rejected():
    with pytest.raises(ValueError):
        constrained.score_constraints('x', [{'kind': 'rhymes', 'value': 'y'}])
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_example_adapters.py -q` — Expected: FAIL, modules missing.

- [ ] **Step 3: Create the classify example**

`examples/__init__.py` and `examples/classify/__init__.py`: empty files.

`examples/classify/task.json`:

```json
{"name": "banking_intents", "objective": "Route a customer message to the single best intent label.",
 "observation_fields": ["text", "labels"],
 "action_space": {"kind": "labels", "labels": ["card_lost", "refund", "balance", "transfer", "other"]},
 "evaluator": {"kind": "exact_match"}, "decisions_per_episode": 1, "latency_budget_ms": 1000,
 "examples": [
  {"observation": {"text": "I can't find my card anywhere"}, "expected": "card_lost"},
  {"observation": {"text": "I want my money back for this charge"}, "expected": "refund"},
  {"observation": {"text": "how much do I have in my account"}, "expected": "balance"},
  {"observation": {"text": "send 50 to my sister"}, "expected": "transfer"},
  {"observation": {"text": "what are your opening hours"}, "expected": "other"}]}
```

`examples/classify/fixture.json` — 40 rows, 8 per label, ids `c01`…`c40`, alternating labels so the first 24 rows contain every label:

```json
[
 {"id": "c01", "text": "I lost my debit card yesterday", "label": "card_lost"},
 {"id": "c02", "text": "please refund the duplicate payment", "label": "refund"},
 {"id": "c03", "text": "what's my current balance", "label": "balance"},
 {"id": "c04", "text": "move 200 to my savings account", "label": "transfer"},
 {"id": "c05", "text": "do you have a branch in Leeds", "label": "other"},
 {"id": "c06", "text": "my card was stolen on the train", "label": "card_lost"},
 {"id": "c07", "text": "I was charged twice, give me my money back", "label": "refund"},
 {"id": "c08", "text": "how much money is left in checking", "label": "balance"},
 {"id": "c09", "text": "wire 1000 to John's account", "label": "transfer"},
 {"id": "c10", "text": "can I change my email address", "label": "other"},
 {"id": "c11", "text": "card missing after the concert", "label": "card_lost"},
 {"id": "c12", "text": "the merchant cancelled but I still paid", "label": "refund"},
 {"id": "c13", "text": "show me my available funds", "label": "balance"},
 {"id": "c14", "text": "pay my landlord 800 tomorrow", "label": "transfer"},
 {"id": "c15", "text": "how do I contact support", "label": "other"},
 {"id": "c16", "text": "I think someone took my card", "label": "card_lost"},
 {"id": "c17", "text": "refund for the subscription I never used", "label": "refund"},
 {"id": "c18", "text": "balance please", "label": "balance"},
 {"id": "c19", "text": "transfer funds between my own accounts", "label": "transfer"},
 {"id": "c20", "text": "is the app down right now", "label": "other"},
 {"id": "c21", "text": "cannot find my card, block it", "label": "card_lost"},
 {"id": "c22", "text": "I want the fee reversed", "label": "refund"},
 {"id": "c23", "text": "what is the balance on my credit card", "label": "balance"},
 {"id": "c24", "text": "send money abroad to my parents", "label": "transfer"},
 {"id": "c25", "text": "update my phone number", "label": "other"},
 {"id": "c26", "text": "my wallet with the card is gone", "label": "card_lost"},
 {"id": "c27", "text": "charged for something I returned", "label": "refund"},
 {"id": "c28", "text": "how much can I spend today", "label": "balance"},
 {"id": "c29", "text": "set up a standing order to my gym", "label": "transfer"},
 {"id": "c30", "text": "where is the nearest ATM", "label": "other"},
 {"id": "c31", "text": "lost card need replacement", "label": "card_lost"},
 {"id": "c32", "text": "the refund never arrived", "label": "refund"},
 {"id": "c33", "text": "tell me my remaining balance", "label": "balance"},
 {"id": "c34", "text": "pay back my friend 30", "label": "transfer"},
 {"id": "c35", "text": "what is your interest rate", "label": "other"},
 {"id": "c36", "text": "card fell out of my pocket", "label": "card_lost"},
 {"id": "c37", "text": "I dispute this transaction, refund it", "label": "refund"},
 {"id": "c38", "text": "account balance after the last payment", "label": "balance"},
 {"id": "c39", "text": "schedule a transfer for Friday", "label": "transfer"},
 {"id": "c40", "text": "how do I close my account", "label": "other"}
]
```

`examples/classify/adapter.py`:

```python
"""Single-step label routing. The label never enters the observation; the labels list does."""
import copy
import json
from pathlib import Path

from auto_jev.runtime import PipelineExecutionError, PipelineRuntime

HERE = Path(__file__).resolve().parent
TRAIN_ROWS = 24


def load(split):
    """Return (task, episodes) for 'train' (first 24 rows) or 'validation' (remaining 16)."""
    task = json.loads((HERE / 'task.json').read_text())
    rows = json.loads((HERE / 'fixture.json').read_text())
    if split == 'train':
        rows = rows[:TRAIN_ROWS]
    elif split == 'validation':
        rows = rows[TRAIN_ROWS:]
    else:
        raise ValueError('split must be train or validation')
    labels = list(task['action_space']['labels'])
    return task, [{**row, 'labels': labels} for row in rows]


def evaluate_episode(spec, episode, jev, llm=None, *, capture_traces=True):
    observation = {'text': episode['text'], 'labels': list(episode['labels'])}
    runtime = PipelineRuntime(spec, jev, llm)
    try:
        decision = runtime.run(observation, memory={})
    except PipelineExecutionError as exc:
        partial = copy.deepcopy(exc.partial_result)
        partial.update(observation=observation, status='error')
        raise PipelineExecutionError(str(exc), partial_result={'episode_id': episode['id'], 'score': None,
                                     'error': str(exc), 'traces': [partial]}, cause=exc.cause, causes=exc.causes) from exc
    output = decision['output']
    record = {**decision, 'observation': observation, 'status': 'ok'}
    if not isinstance(output, str) or output not in episode['labels']:
        record.update(status='error', error='Policy returned a value outside the label set')
        error = ValueError('Policy returned a value outside the label set')
        raise PipelineExecutionError(str(error), partial_result={'episode_id': episode['id'], 'score': None,
                                     'error': str(error), 'traces': [record]}, cause=error) from error
    return {'episode_id': episode['id'], 'task_id': 'banking_intents', 'status': 'completed',
            'score': 1.0 if output == episode['label'] else 0.0, 'output': output,
            'traces': [record] if capture_traces else []}
```

- [ ] **Step 4: Create the constrained_text example**

`examples/constrained_text/__init__.py`: empty.

`examples/constrained_text/task.json`:

```json
{"name": "constrained_reply", "objective": "Write a short customer reply that satisfies every listed constraint.",
 "observation_fields": ["instruction", "constraints"],
 "action_space": {"kind": "open"}, "evaluator": {"kind": "code_check"},
 "decisions_per_episode": 1, "latency_budget_ms": null,
 "output_schema": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string", "minLength": 1}}},
 "examples": [
  {"observation": {"instruction": "Apologise for a late delivery.", "constraints": [{"kind": "max_words", "value": 25}, {"kind": "must_include", "value": "sorry"}]},
   "expected": "We are sorry your delivery arrived late; a credit has been added to your account."},
  {"observation": {"instruction": "Confirm the refund was issued.", "constraints": [{"kind": "must_include", "value": "refund"}, {"kind": "ends_with", "value": "."}]},
   "expected": "Your refund has been issued and will appear within five business days."}]}
```

`examples/constrained_text/fixture.json` — 20 rows, ids `r01`…`r20`; first 12 train, last 8 validation. Each row: `{"id", "instruction", "constraints"}`. Write these exactly:

```json
[
 {"id": "r01", "instruction": "Apologise for a late delivery.", "constraints": [{"kind": "max_words", "value": 25}, {"kind": "must_include", "value": "sorry"}]},
 {"id": "r02", "instruction": "Confirm a refund was issued.", "constraints": [{"kind": "must_include", "value": "refund"}, {"kind": "ends_with", "value": "."}]},
 {"id": "r03", "instruction": "Thank a customer for waiting.", "constraints": [{"kind": "max_words", "value": 15}, {"kind": "must_include", "value": "thank"}]},
 {"id": "r04", "instruction": "Explain that the office is closed on Sunday.", "constraints": [{"kind": "must_include", "value": "Sunday"}, {"kind": "must_not_include", "value": "unfortunately"}]},
 {"id": "r05", "instruction": "Ask the customer for their order number.", "constraints": [{"kind": "must_include", "value": "order number"}, {"kind": "ends_with", "value": "?"}]},
 {"id": "r06", "instruction": "Confirm an appointment for Tuesday at 10am.", "constraints": [{"kind": "must_include", "value": "Tuesday"}, {"kind": "must_include", "value": "10"}, {"kind": "max_words", "value": 20}]},
 {"id": "r07", "instruction": "Decline a discount request politely.", "constraints": [{"kind": "must_not_include", "value": "no"}, {"kind": "min_words", "value": 8}]},
 {"id": "r08", "instruction": "Tell the customer their card has been blocked.", "constraints": [{"kind": "must_include", "value": "blocked"}, {"kind": "max_words", "value": 18}]},
 {"id": "r09", "instruction": "Wish the customer a good weekend.", "constraints": [{"kind": "must_include", "value": "weekend"}, {"kind": "max_words", "value": 10}]},
 {"id": "r10", "instruction": "Explain that a password reset link was sent.", "constraints": [{"kind": "must_include", "value": "link"}, {"kind": "must_include", "value": "email"}]},
 {"id": "r11", "instruction": "Confirm a subscription was cancelled.", "constraints": [{"kind": "must_include", "value": "cancelled"}, {"kind": "ends_with", "value": "."}]},
 {"id": "r12", "instruction": "Ask the customer to try restarting the app.", "constraints": [{"kind": "must_include", "value": "restart"}, {"kind": "max_words", "value": 20}]},
 {"id": "r13", "instruction": "Apologise for a billing error.", "constraints": [{"kind": "must_include", "value": "sorry"}, {"kind": "must_include", "value": "billing"}]},
 {"id": "r14", "instruction": "Confirm the address change.", "constraints": [{"kind": "must_include", "value": "address"}, {"kind": "max_words", "value": 15}]},
 {"id": "r15", "instruction": "Tell the customer a technician will call tomorrow.", "constraints": [{"kind": "must_include", "value": "tomorrow"}, {"kind": "must_include", "value": "call"}]},
 {"id": "r16", "instruction": "Thank the customer for their feedback.", "constraints": [{"kind": "must_include", "value": "feedback"}, {"kind": "ends_with", "value": "."}]},
 {"id": "r17", "instruction": "Explain that the item is out of stock.", "constraints": [{"kind": "must_include", "value": "stock"}, {"kind": "must_not_include", "value": "never"}]},
 {"id": "r18", "instruction": "Confirm the delivery date is Friday.", "constraints": [{"kind": "must_include", "value": "Friday"}, {"kind": "max_words", "value": 12}]},
 {"id": "r19", "instruction": "Ask whether the customer wants a replacement or a refund.", "constraints": [{"kind": "must_include", "value": "replacement"}, {"kind": "must_include", "value": "refund"}, {"kind": "ends_with", "value": "?"}]},
 {"id": "r20", "instruction": "Close the conversation warmly.", "constraints": [{"kind": "min_words", "value": 5}, {"kind": "max_words", "value": 15}]}
]
```

`examples/constrained_text/adapter.py`:

```python
"""Open-ended reply generation scored by deterministic constraint checks."""
import copy
import json
from pathlib import Path

from auto_jev.runtime import PipelineExecutionError, PipelineRuntime

HERE = Path(__file__).resolve().parent
TRAIN_ROWS = 12
KINDS = ('max_words', 'min_words', 'must_include', 'must_not_include', 'ends_with')


def load(split):
    task = json.loads((HERE / 'task.json').read_text())
    rows = json.loads((HERE / 'fixture.json').read_text())
    if split == 'train':
        rows = rows[:TRAIN_ROWS]
    elif split == 'validation':
        rows = rows[TRAIN_ROWS:]
    else:
        raise ValueError('split must be train or validation')
    return task, rows


def score_constraints(text, constraints):
    """Fraction of constraints satisfied; unknown kinds are a configuration error."""
    if not constraints:
        return 1.0
    words = text.split()
    passed = 0
    for constraint in constraints:
        kind, value = constraint['kind'], constraint['value']
        if kind not in KINDS:
            raise ValueError(f'unknown constraint kind {kind!r}')
        if kind == 'max_words':
            passed += len(words) <= value
        elif kind == 'min_words':
            passed += len(words) >= value
        elif kind == 'must_include':
            passed += str(value).lower() in text.lower()
        elif kind == 'must_not_include':
            passed += str(value).lower() not in text.lower()
        else:
            passed += text.rstrip().endswith(str(value))
    return passed / len(constraints)


def evaluate_episode(spec, episode, jev, llm=None, *, capture_traces=True):
    observation = {'instruction': episode['instruction'], 'constraints': copy.deepcopy(episode['constraints'])}
    runtime = PipelineRuntime(spec, jev, llm)
    try:
        decision = runtime.run(observation, memory={})
    except PipelineExecutionError as exc:
        partial = copy.deepcopy(exc.partial_result)
        partial.update(observation=observation, status='error')
        raise PipelineExecutionError(str(exc), partial_result={'episode_id': episode['id'], 'score': None,
                                     'error': str(exc), 'traces': [partial]}, cause=exc.cause, causes=exc.causes) from exc
    output = decision['output']
    record = {**decision, 'observation': observation, 'status': 'ok'}
    if not isinstance(output, dict) or not isinstance(output.get('text'), str) or not output['text'].strip():
        record.update(status='error', error='Policy must return {"text": nonempty string}')
        error = ValueError('Policy must return {"text": nonempty string}')
        raise PipelineExecutionError(str(error), partial_result={'episode_id': episode['id'], 'score': None,
                                     'error': str(error), 'traces': [record]}, cause=error) from error
    return {'episode_id': episode['id'], 'task_id': 'constrained_reply', 'status': 'completed',
            'score': score_constraints(output['text'], episode['constraints']), 'output': output,
            'traces': [record] if capture_traces else []}
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_example_adapters.py -q` — Expected: PASS.

- [ ] **Step 6: Commit** — `git add examples tests/test_example_adapters.py && git commit -m "feat: classify and constrained_text example adapters with fixtures"`

---

### Task 8: `run_routed_evolution` and end-to-end acceptance

**Files:**
- Create: `auto_jev/routed.py`
- Test: `tests/test_routed.py`

**Interfaces:**
- Consumes: `triage.triage`, `triage.seed_for_route`, `triage.route_of_spec`, `triage.validate_task`, `evolution.run_evolution`, `storage.RunStore`, `storage.atomic_write_json`, `storage.now_iso`. Task 9 later adds `triage_store.current_questions/record/improve` calls; Task 8 leaves `# triage_store hook` comments where they go.
- Produces: `run_routed_evolution(task, train, validation, *, jev, llm, proposer, store, evaluator, task_id, route=None, seed_pipeline=None, budget=None, evolution_rounds=None, max_metric_calls=None, reflection_batch_size=None, seed=0, task_context=None, task_contract=None, triage_questions=None) -> summary`, `SEED_ROUTE_OF = {'settled':'jev','open_ended':'llm','hybrid':'hybrid'}`.

- [ ] **Step 1: Write the failing tests**

```python
import json

import pytest

from auto_jev.evolution import freeze_run
from auto_jev.frozen import evaluate_frozen
from auto_jev.llm import LLMClient
from auto_jev.routed import run_routed_evolution
from auto_jev.storage import RunStore
from examples.classify import adapter as classify
from examples.constrained_text import adapter as constrained
from tests.support import ScriptedJev


def sandbox_ok():
    from auto_jev.python_nodes import PythonSandboxUnavailable, _sandbox_command
    try:
        _sandbox_command(); return True
    except PythonSandboxUnavailable:
        return False


pytestmark = pytest.mark.skipif(not sandbox_ok(), reason='seeds use Python nodes')


def echo_proposer(prompt):
    payload = json.loads(prompt)
    assert payload['task']['context']['route'] in ('settled', 'open_ended', 'hybrid')
    assert 'probabilities' in payload['task']['context']['triage'] and 'response' not in payload['task']['context']['triage']
    return json.dumps(payload['parent'])


def test_settled_task_routes_to_jev_runs_once_and_freezes(tmp_path):
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    jev = ScriptedJev({'route': {'settled': 0.9, 'open_ended': 0.05, 'hybrid': 0.05},
                       'action': {'card_lost': 0.2, 'refund': 0.6, 'balance': 0.1, 'transfer': 0.05, 'other': 0.05}})
    store = RunStore(tmp_path)
    summary = run_routed_evolution(task, train[:4], validation[:4], jev=jev, llm=LLMClient(kind='mock'),
                                   proposer=echo_proposer, store=store, evaluator=classify.evaluate_episode,
                                   task_id='banking', evolution_rounds=1, reflection_batch_size=1,
                                   budget={'latency_ms_p95': 60000})
    assert summary['decision']['route'] == 'settled' and summary['route'] == 'settled'
    assert summary['route_source'] == 'triage' and summary['seed_route'] == 'jev' and summary['final_route'] == 'jev'
    assert len(store.list_runs()) == 1 and summary['run_id'] == store.list_runs()[0]['run_id']
    assert store.get_run(summary['run_id'])['config']['task_context']['route'] == 'settled'
    saved = json.loads((tmp_path / 'routed' / summary['routed_id'] / 'summary.json').read_text())
    assert saved['run_id'] == summary['run_id'] and saved['decision']['probabilities']['settled'] == 0.9
    artifact = freeze_run(store, summary['run_id'])
    outputs = evaluate_frozen(artifact, validation[:2], jev, evaluator=classify.evaluate_episode, llm=LLMClient(kind='mock'))
    assert all(o['score'] in (0.0, 1.0) for o in outputs)


def test_open_ended_task_runs_llm_seed(tmp_path):
    task, train = constrained.load('train')
    _, validation = constrained.load('validation')
    jev = ScriptedJev({'route': {'settled': 0.1, 'open_ended': 0.5, 'hybrid': 0.4}})
    llm = LLMClient(kind='mock', model='mock-llm', mock_responder=lambda p, s: json.dumps({'text': 'We are sorry and thank you; your refund is issued.'}))
    summary = run_routed_evolution(task, train[:3], validation[:3], jev=jev, llm=llm, proposer=echo_proposer,
                                   store=RunStore(tmp_path), evaluator=constrained.evaluate_episode,
                                   task_id='reply', evolution_rounds=1, reflection_batch_size=1)
    assert summary['route'] == 'open_ended' and summary['decision']['confidence'] == 0.5
    assert summary['seed_route'] == 'llm' and summary['final_route'] == 'llm' and summary['seed_llm_model'] == 'mock-llm'
    assert 0.0 <= summary['best_score'] <= 1.0


def test_route_override_runs_hybrid_seed(tmp_path):
    task, train = constrained.load('train')
    _, validation = constrained.load('validation')
    jev = ScriptedJev({'route': {'settled': 0.1, 'open_ended': 0.5, 'hybrid': 0.4}, 'best': {'c0': 0.2, 'c1': 0.7, 'c2': 0.1}})

    def responder(prompt, schema):
        return json.dumps({'candidates': ['sorry', 'We are sorry, thank you, refund issued.', 'x']})

    summary = run_routed_evolution(task, train[:2], validation[:2], jev=jev, llm=LLMClient(kind='mock', mock_responder=responder),
                                   proposer=echo_proposer, store=RunStore(tmp_path), evaluator=constrained.evaluate_episode,
                                   task_id='reply', route='hybrid', evolution_rounds=1, reflection_batch_size=1)
    assert summary['decision']['route'] == 'open_ended' and summary['route'] == 'hybrid'
    assert summary['route_source'] == 'override' and summary['seed_route'] == 'hybrid' and summary['final_route'] == 'hybrid'


def test_seed_pipeline_override_replaces_template(tmp_path):
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    jev = ScriptedJev({'route': {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0}})
    override = {'version': 4, 'name': 'const', 'jev_model': 'typesafe-ai/jev', 'nodes': [], 'output': "'refund'"}
    summary = run_routed_evolution(task, train[:2], validation[:2], jev=jev, llm=LLMClient(kind='mock'),
                                   proposer=echo_proposer, store=RunStore(tmp_path), evaluator=classify.evaluate_episode,
                                   task_id='banking', evolution_rounds=1, reflection_batch_size=1, seed_pipeline=override)
    assert summary['seed_route'] == 'jev' and summary['final_route'] == 'code'


def test_unknown_route_override_is_rejected(tmp_path):
    task, train = classify.load('train')
    with pytest.raises(ValueError, match='route'):
        run_routed_evolution(task, train[:1], train[1:2], jev=ScriptedJev(), llm=LLMClient(kind='mock'),
                             proposer=echo_proposer, store=RunStore(tmp_path), evaluator=classify.evaluate_episode,
                             task_id='banking', route='teleport', evolution_rounds=1, reflection_batch_size=1)
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_routed.py -q` — Expected: FAIL, module not found.

- [ ] **Step 3: Implement `auto_jev/routed.py`**

```python
"""Level 1 orchestration: probe → Jev triage → one seed → one run_evolution. Re-routing happens inside evolution."""
import copy
import time
import uuid

from .evolution import run_evolution
from .spec import spec_hash, validate_spec
from .storage import RunStore, atomic_write_json, now_iso
from .triage import ROUTES, route_of_spec, seed_for_route, triage, validate_task

SEED_ROUTE_OF = {'settled': 'jev', 'open_ended': 'llm', 'hybrid': 'hybrid'}


def _routed_dir(store, task_id):
    ident = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{task_id}-{uuid.uuid4().hex[:6]}"
    path = store.root / 'routed' / ident
    path.mkdir(parents=True, exist_ok=False)
    return ident, path


def run_routed_evolution(task, train, validation, *, jev, llm, proposer, store, evaluator, task_id, route=None,
                         seed_pipeline=None, budget=None, evolution_rounds=None, max_metric_calls=None,
                         reflection_batch_size=None, seed=0, task_context=None, task_contract=None, triage_questions=None):
    task = validate_task(task)
    if route is not None and route not in ROUTES:
        raise ValueError('route must be one of ' + ', '.join(ROUTES))
    store = store or RunStore()
    # triage_store hook (Task 9): questions default to the stored, improved version.
    decision = triage(task, jev, questions=triage_questions, jev_model=getattr(jev, 'model', 'typesafe-ai/jev'))
    route_source = 'override' if route is not None else 'triage'
    route = route or decision['route']
    routed_id, path = _routed_dir(store, task_id)
    atomic_write_json(path / 'decision.json', {**decision, 'route_source': route_source, 'chosen_route': route})
    spec = copy.deepcopy(seed_pipeline) if seed_pipeline is not None else seed_for_route(route, task)
    llm_model = getattr(llm, 'model', None)
    if llm_model and any(node.get('kind') == 'llm' for node in spec['nodes']):
        spec['llm_model'] = llm_model
    spec = validate_spec(spec)
    context = copy.deepcopy(task_context) if task_context else {}
    context.setdefault('objective', task['objective'])
    context.update(route=route, route_source=route_source,
                   triage={key: decision[key] for key in ('route', 'probabilities', 'confidence', 'evidence')})
    summary = run_evolution(train, validation, jev=jev, llm=llm, proposer=proposer, store=store,
                            run_name=f'{task_id}-{route}', evaluator=evaluator, task_id=task_id, costs={},
                            seed_pipeline=spec, task_context=context, task_contract=task_contract, budget=budget,
                            evolution_rounds=evolution_rounds, max_metric_calls=max_metric_calls,
                            reflection_batch_size=reflection_batch_size, seed=seed)
    best = store.get_candidate(summary['run_id'], summary['best_hash'])['spec']
    result = {**summary, 'routed_id': routed_id, 'task_id': task_id, 'created_at': now_iso(), 'decision': decision,
              'route': route, 'route_source': route_source, 'seed_hash': spec_hash(spec), 'seed_route': SEED_ROUTE_OF[route],
              'seed_llm_model': spec.get('llm_model'), 'final_route': route_of_spec(best), 'budget': budget}
    # triage_store hook (Task 9): record history and attempt one improvement here.
    atomic_write_json(path / 'summary.json', result)
    return result
```

Check that `RunStore` exposes `.root` (a `Path`); if the attribute is named differently in `storage.py`, use that name.

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_routed.py -q` — Expected: PASS.

- [ ] **Step 5: Run the whole suite** — `uv run pytest -q` — Expected: PASS (darwin-only tests skipped).

- [ ] **Step 6: Commit** — `git add auto_jev/routed.py tests/test_routed.py && git commit -m "feat: run_routed_evolution triages, seeds one harness and runs evolution once"`

---

### Task 9: Triage store — history and replay-validated criteria improvement

**Files:**
- Create: `auto_jev/triage_store.py`
- Modify: `auto_jev/routed.py` (the two `# triage_store hook` sites), `auto_jev/triage.py` (export `TARGET_OF`)
- Test: `tests/test_triage_store.py`, extend `tests/test_routed.py`

**Interfaces:**
- Consumes: `triage.DEFAULT_TRIAGE_QUESTIONS`, `triage.ROUTES`, `spec.validate_questions`, `storage.atomic_write_json`, `storage.read_json`, `storage.append_jsonl`, `storage.read_jsonl`, `storage.now_iso`, `reflection.atomic_text`.
- Produces: `triage.TARGET_OF = {'jev':'settled','llm':'open_ended','hybrid':'hybrid','code':'settled'}`; `triage_store.current_questions(store) -> (questions, version)`; `triage_store.record(store, summary) -> record`; `triage_store.replay_agreement(questions, history, jev) -> float`; `triage_store.improve(store, jev, proposer, *, min_history=2) -> result`; `run_routed_evolution` summary gains `triage_improvement` and `questions_version`.

- [ ] **Step 1: Write the failing tests (`tests/test_triage_store.py`)**

```python
import json

import pytest

from auto_jev import triage_store
from auto_jev.storage import RunStore
from auto_jev.triage import DEFAULT_TRIAGE_QUESTIONS, TARGET_OF
from tests.support import ScriptedJev


def summary(routed_id, task_id, state, probs, final_route, score=0.5, version=0):
    return {'routed_id': routed_id, 'task_id': task_id, 'run_id': 'r-' + routed_id, 'best_score': score,
            'route': max(probs, key=probs.get), 'route_source': 'triage', 'final_route': final_route,
            'questions_version': version,
            'decision': {'route': max(probs, key=probs.get), 'probabilities': probs, 'confidence': max(probs.values()),
                         'evidence': {'declared_action_kind': 'labels'}, 'questions': DEFAULT_TRIAGE_QUESTIONS,
                         'state': state, 'response': {}},
            'task': {'name': task_id}}


class ReplayJev(ScriptedJev):
    """Answers the route question by looking for a marker word in the criteria text."""
    def __init__(self, marker_to_route):
        super().__init__()
        self.marker_to_route = marker_to_route

    def judge(self, state, questions, model=None):
        self.requests.append({'state': state, 'questions': questions})
        text = json.dumps(questions['route']['criteria'])
        route = next((r for marker, r in self.marker_to_route.items() if marker in text and state.get('marker') == marker), 'settled')
        probs = {r: 0.0 for r in ('settled', 'open_ended', 'hybrid')}; probs[route] = 1.0
        return {'answers': {'route': {'type': 'choice', 'choice': route, 'probabilities': probs, 'confidence': 1.0},
                            'code_computes_facts': {'type': 'noul', 'noul': 0.5}, 'needs_multistep': {'type': 'noul', 'noul': 0.5}},
                'model': 'replay', 'usage': {}, 'cache_hit': False, 'elapsed_ms': 1.0, 'client_metadata': {}}


def test_default_questions_when_nothing_stored(tmp_path):
    questions, version = triage_store.current_questions(RunStore(tmp_path))
    assert questions == DEFAULT_TRIAGE_QUESTIONS and version == 0


def test_record_maps_final_route_to_target(tmp_path):
    store = RunStore(tmp_path)
    rec = triage_store.record(store, summary('a', 'ta', {'marker': 'alpha'}, {'settled': 0.9, 'open_ended': 0.1, 'hybrid': 0.0}, 'llm'))
    assert rec['target'] == TARGET_OF['llm'] == 'open_ended' and rec['triage_route'] == 'settled'
    assert triage_store.history(store) == [rec]


def test_improve_skips_below_min_history(tmp_path):
    store = RunStore(tmp_path)
    triage_store.record(store, summary('a', 'ta', {'marker': 'alpha'}, {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0}, 'jev'))
    result = triage_store.improve(store, ScriptedJev(), lambda prompt: '{}', min_history=2)
    assert result['status'] == 'skipped' and triage_store.current_questions(store)[1] == 0


def two_records(store):
    triage_store.record(store, summary('a', 'ta', {'marker': 'alpha'}, {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0}, 'llm'))
    triage_store.record(store, summary('b', 'tb', {'marker': 'beta'}, {'settled': 1.0, 'open_ended': 0.0, 'hybrid': 0.0}, 'hybrid'))


def better_proposal(prompt):
    payload = json.loads(prompt)
    questions = json.loads(json.dumps(payload['current_questions']))
    questions['route']['criteria']['open_ended'] += ' alpha'
    questions['route']['criteria']['hybrid'] += ' beta'
    return json.dumps(questions)


def test_better_proposal_is_accepted_and_versioned(tmp_path):
    store = RunStore(tmp_path)
    two_records(store)
    jev = ReplayJev({'alpha': 'open_ended', 'beta': 'hybrid'})
    result = triage_store.improve(store, jev, better_proposal, min_history=2)
    assert result['status'] == 'accepted' and result['agreement_before'] == 0.0 and result['agreement_after'] == 1.0
    questions, version = triage_store.current_questions(store)
    assert version == 1 and 'alpha' in questions['route']['criteria']['open_ended']
    assert (tmp_path / 'triage' / 'questions.json').exists()
    assert len(list((tmp_path / 'triage' / 'reflections').glob('*.prompt.json'))) == 1
    assert triage_store.improvements(store)[-1]['accepted'] is True


def test_worse_or_equal_proposal_is_rejected_and_file_unchanged(tmp_path):
    store = RunStore(tmp_path)
    two_records(store)
    jev = ReplayJev({'alpha': 'open_ended', 'beta': 'hybrid'})
    same = lambda prompt: json.dumps(json.loads(prompt)['current_questions'])
    result = triage_store.improve(store, jev, same, min_history=2)
    assert result['status'] == 'rejected' and 'not better' in result['reason']
    assert triage_store.current_questions(store)[1] == 0


def test_invalid_proposal_is_rejected(tmp_path):
    store = RunStore(tmp_path)
    two_records(store)
    result = triage_store.improve(store, ScriptedJev(), lambda prompt: '{"route": {"type": "choice"}}', min_history=2)
    assert result['status'] == 'rejected'
    wrong_keys = lambda prompt: json.dumps({**json.loads(prompt)['current_questions'], 'route': {'type': 'choice', 'instructions': 'x', 'criteria': {'a': 'b', 'c': 'd', 'e': 'f'}}})
    assert triage_store.improve(store, ScriptedJev(), wrong_keys, min_history=2)['status'] == 'rejected'


def test_proposer_error_is_recorded_not_raised(tmp_path):
    store = RunStore(tmp_path)
    two_records(store)
    def broken(prompt):
        raise RuntimeError('provider down')
    result = triage_store.improve(store, ScriptedJev(), broken, min_history=2)
    assert result['status'] == 'error' and 'provider down' in result['reason']
```

Also append to `tests/test_routed.py`:

```python
def test_routed_run_records_history_and_attempts_improvement(tmp_path):
    from auto_jev import triage_store
    task, train = classify.load('train')
    _, validation = classify.load('validation')
    jev = ScriptedJev({'route': {'settled': 0.9, 'open_ended': 0.05, 'hybrid': 0.05},
                       'action': {'card_lost': 0.2, 'refund': 0.6, 'balance': 0.1, 'transfer': 0.05, 'other': 0.05}})
    store = RunStore(tmp_path)
    first = run_routed_evolution(task, train[:2], validation[:2], jev=jev, llm=LLMClient(kind='mock'), proposer=echo_proposer,
                                 store=store, evaluator=classify.evaluate_episode, task_id='banking', evolution_rounds=1, reflection_batch_size=1)
    assert first['questions_version'] == 0 and first['triage_improvement']['status'] == 'skipped'
    second = run_routed_evolution(task, train[2:4], validation[2:4], jev=jev, llm=LLMClient(kind='mock'), proposer=echo_proposer,
                                  store=store, evaluator=classify.evaluate_episode, task_id='banking2', evolution_rounds=1, reflection_batch_size=1)
    assert len(triage_store.history(store)) == 2
    assert second['triage_improvement']['status'] in ('accepted', 'rejected', 'error')
```

(`echo_proposer` returns the parent spec for evolution prompts; for the triage prompt — which has no `parent` key — it must return the current questions. Update `echo_proposer` in `tests/test_routed.py`:)

```python
def echo_proposer(prompt):
    payload = json.loads(prompt)
    if 'current_questions' in payload:
        return json.dumps(payload['current_questions'])
    assert payload['task']['context']['route'] in ('settled', 'open_ended', 'hybrid')
    assert 'probabilities' in payload['task']['context']['triage'] and 'response' not in payload['task']['context']['triage']
    return json.dumps(payload['parent'])
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_triage_store.py -q` — Expected: FAIL, module not found.

- [ ] **Step 3: Add `TARGET_OF` and `state` to `auto_jev/triage.py`**

After `ROUTES` add `TARGET_OF = {'jev': 'settled', 'llm': 'open_ended', 'hybrid': 'hybrid', 'code': 'settled'}`. In `triage()`, include `'state': state` in the returned decision (the exact object sent to Jev, needed for replay).

- [ ] **Step 4: Implement `auto_jev/triage_store.py`**

```python
"""Persisted triage questions with per-run history and replay-validated improvement."""
import copy
import hashlib
import json
from pathlib import Path

from .reflection import atomic_text
from .spec import validate_questions
from .storage import append_jsonl, atomic_write_json, now_iso, read_json, read_jsonl
from .triage import DEFAULT_TRIAGE_QUESTIONS, ROUTES, TARGET_OF

INSTRUCTION = ('Improve the triage question map used to route tasks before harness authoring. '
               'Return ONLY a complete JSON question map with the same three question ids (route, code_computes_facts, needs_multistep). '
               'The route question is a choice whose criteria keys are exactly settled, open_ended and hybrid; rewrite the '
               'instructions and criteria so that, given each history record\'s state, the route answer matches its target. '
               'Targets are the forms the evolved harnesses actually ended up as. Do not reference task names; write general criteria.')


def _dir(store):
    path = Path(store.root) / 'triage'
    path.mkdir(parents=True, exist_ok=True)
    return path


def current_questions(store):
    stored = read_json(_dir(store) / 'questions.json', None)
    if not stored:
        return copy.deepcopy(DEFAULT_TRIAGE_QUESTIONS), 0
    return copy.deepcopy(stored['questions']), int(stored['version'])


def history(store):
    return read_jsonl(_dir(store) / 'history.jsonl')


def improvements(store):
    return read_jsonl(_dir(store) / 'improvements.jsonl')


def record(store, summary):
    decision = summary['decision']
    entry = {'routed_id': summary['routed_id'], 'task_id': summary['task_id'],
             'task_name': (summary.get('task') or {}).get('name'), 'run_id': summary['run_id'],
             'state': copy.deepcopy(decision['state']), 'probabilities': decision['probabilities'],
             'triage_route': decision['route'], 'chosen_route': summary['route'], 'route_source': summary['route_source'],
             'final_route': summary['final_route'], 'target': TARGET_OF[summary['final_route']],
             'best_score': summary['best_score'], 'questions_version': summary.get('questions_version', 0),
             'recorded_at': now_iso()}
    append_jsonl(_dir(store) / 'history.jsonl', entry)
    return entry


def replay_agreement(questions, records, jev):
    if not records:
        return 0.0
    hits = 0
    for entry in records:
        answer = jev.judge(copy.deepcopy(entry['state']), copy.deepcopy(questions))['answers']['route']
        probabilities = answer.get('probabilities') or {}
        choice = max(probabilities, key=probabilities.get) if probabilities else answer.get('choice')
        hits += choice == entry['target']
    return hits / len(records)


def _validate_proposal(text):
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    proposed = validate_questions(json.loads(text), strict=True)
    if set(proposed) != set(DEFAULT_TRIAGE_QUESTIONS):
        raise ValueError('proposal must keep the three triage question ids')
    if proposed['route']['type'] != 'choice' or set(proposed['route']['criteria']) != set(ROUTES):
        raise ValueError('route criteria keys must be exactly settled, open_ended, hybrid')
    return proposed


def improve(store, jev, proposer, *, min_history=2):
    records = history(store)
    current, version = current_questions(store)
    if len(records) < min_history:
        result = {'status': 'skipped', 'reason': f'history has {len(records)} records; need {min_history}', 'version': version}
        append_jsonl(_dir(store) / 'improvements.jsonl', {**result, 'accepted': False, 'at': now_iso()})
        return result
    payload = {'instruction': INSTRUCTION, 'vocabulary': list(ROUTES), 'current_questions': current,
               'history': [{k: entry[k] for k in ('state', 'probabilities', 'triage_route', 'target', 'best_score')} for entry in records]}
    prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    sha = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    reflections = _dir(store) / 'reflections'
    number = len(improvements(store)) + 1
    atomic_text(reflections / f'{number:04d}-{sha}.prompt.json', prompt)
    outcome = {'version': version, 'prompt_sha256': sha, 'history_size': len(records), 'at': now_iso()}
    try:
        text = proposer(prompt).strip()
        atomic_text(reflections / f'{number:04d}-{sha}.response.txt', text)
        proposed = _validate_proposal(text)
    except Exception as exc:
        status = 'error' if not isinstance(exc, (ValueError, TypeError, KeyError)) else 'rejected'
        result = {**outcome, 'status': status, 'reason': f'{type(exc).__name__}: {exc}'[:500]}
        append_jsonl(_dir(store) / 'improvements.jsonl', {**result, 'accepted': False})
        return result
    before = replay_agreement(current, records, jev)
    after = replay_agreement(proposed, records, jev)
    accepted = after > before
    result = {**outcome, 'status': 'accepted' if accepted else 'rejected', 'agreement_before': before, 'agreement_after': after,
              'proposal_sha256': hashlib.sha256(json.dumps(proposed, sort_keys=True).encode()).hexdigest()}
    if accepted:
        version += 1
        atomic_write_json(_dir(store) / 'questions.json', {'version': version, 'questions': proposed, 'accepted_at': now_iso(),
                                                             'agreement': after, 'history_size': len(records)})
        result['version'] = version
    else:
        result['reason'] = f'replay agreement {after} is not better than {before}'
    append_jsonl(_dir(store) / 'improvements.jsonl', {**result, 'accepted': accepted})
    return result
```

Check `storage.append_jsonl` / `read_jsonl` signatures in the vendored `storage.py` (they exist at lines ~147–160 upstream and take a `Path`).

- [ ] **Step 5: Wire the hooks in `auto_jev/routed.py`**

Add `from . import triage_store`. Replace the first hook with:

```python
    stored_questions, questions_version = triage_store.current_questions(store)
    decision = triage(task, jev, questions=triage_questions or stored_questions, jev_model=getattr(jev, 'model', 'typesafe-ai/jev'))
```

Add `'questions_version': questions_version if triage_questions is None else None, 'task': {'name': task['name']}` to `result`. Replace the second hook with:

```python
    triage_store.record(store, result)
    result['triage_improvement'] = triage_store.improve(store, jev, proposer)
```

- [ ] **Step 6: Run** — `uv run pytest tests/test_triage_store.py tests/test_routed.py tests/test_triage.py -q` — Expected: PASS.

- [ ] **Step 7: Commit** — `git add auto_jev/triage_store.py auto_jev/triage.py auto_jev/routed.py tests/test_triage_store.py tests/test_routed.py && git commit -m "feat: replay-validated triage criteria improvement after every routed run"`

---

### Task 10: Documentation

**Files:**
- Create: `docs/routed-harness.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/routed-harness.md`**

```markdown
# Routed harness (Level 1)

## Flow

1. `triage.probe(task)` computes deterministic evidence from a `TaskDescription` (declared action kind,
   label count, whether examples carry a legal-action catalog, evaluator exactness, expected output shape).
2. `triage.triage(task, jev)` sends the evidence and up to five clipped examples to Jev with
   `DEFAULT_TRIAGE_QUESTIONS`. The `route` is the argmax of the choice distribution; the distribution
   is recorded so a low-confidence call is visible, and the caller may pass `route=` to override.
3. `triage.seed_for_route(route, task)` returns a version-4 spec: `settled` → python catalog + jev pick +
   python decision; `open_ended` → python prompt + llm draft + python validate; `hybrid` → llm propose k
   candidates → python criteria → jev select → python decision.
4. `routed.run_routed_evolution(...)` runs `evolution.run_evolution` exactly once with that seed. A wrong
   triage is corrected inside evolution: the reflection model may swap node kinds. `final_route` in the
   summary reports what the selected candidate ended up as. Artifacts: `runs/routed/<id>/{decision,summary}.json`.
5. After the run, `triage_store.record` appends `{state, probabilities, triage_route, final_route → target}` to
   `runs/triage/history.jsonl` and `triage_store.improve` asks the reflection model for a better question map.
   Old and new maps are replayed through Jev on the whole history; the proposal is accepted only if its
   agreement with the targets is strictly higher. Accepted maps are versioned in `runs/triage/questions.json`.

## Version 4 specs

Adds `llm` nodes (`prompt_expression`, optional `output_schema`, optional `max_tokens`) and top-level
`llm_model`. Reflection may swap node kinds and insert generate-then-select structures; it may not change
`jev_model`, `llm_model` or the evaluator.

## Budgets

`run_evolution(..., budget={'latency_ms_p95': ..., 'cost_usd_per_episode': ...})`. Usage is derived from
complete traces (`budget.episode_usage`). A scored episode over budget becomes an invalid candidate
(`status='over_budget'`, score `-1e9`). Unpriced calls are counted, not guessed.

## Linux

Functional Python nodes run under `bwrap`: empty root, read-only `/usr` and the interpreter prefix,
no network, no host environment, `RLIMIT_AS`/`RLIMIT_CPU`/wall-time limits. Missing `bwrap` or refused
user namespaces raise `PythonSandboxUnavailable` (an infrastructure error, never a candidate score).

## Examples

    uv run python -c "from examples.classify import adapter as a; print(a.load('train')[0]['name'])"

`examples/classify` (settled, labels) and `examples/constrained_text` (open-ended, code-checked) run
end-to-end with mock providers in `tests/test_routed.py`.
```

- [ ] **Step 2: Add a "Routing" section to `README.md` linking `docs/routed-harness.md` and the artifact URL.**

- [ ] **Step 3: Commit** — `git add docs/routed-harness.md README.md && git commit -m "docs: routed harness Level 1"`

---

## Self-review notes

- Spec coverage: §1 → Task 1; §2 → Tasks 2–3; §3 → Task 4; §4 → Task 5; §5 → Task 6; §6 → Task 8; §7 → Task 7; §8 → Task 9; acceptance → Task 8/9 tests + Task 5 frozen test.
- Review Focus items 1–5 are pinned in Tasks 4, 6, 5, 5, 1 respectively. Multi-seed / cross-run selection was removed on request: one triage, one seed, one evolution.
- Names used across tasks: `ScriptedJev` (Task 0), `LLMClient.complete` (3, used in 4/6/8), `PipelineRuntime(spec, jev, llm=None)` (4, used in 6/7/8), `run_evolution(budget=, llm=)` and `call_evaluator` (5, used in 8 and frozen), `evaluate_frozen(llm=)` (5, used in 8), `triage`/`seed_for_route`/`route_of_spec`/`validate_task`/`ROUTES` (6, used in 8), `load(split)` adapters (7, used in 8).
