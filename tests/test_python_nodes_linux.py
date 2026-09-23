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
    for flag in ('--unshare-all', '--die-with-parent', '--new-session', '--proc', '--dev', '--tmpfs'):
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
    stream = SimpleNamespace(close=lambda: None, closed=True)
    process = SimpleNamespace(stdin=stream, stdout=stream, stderr=stream,
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
        evaluate_python(source('big = [0] * (200 * 1024 * 1024)\nreturn len(big)'), copy.deepcopy(CONTEXT))
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
    # bwrap does not deny fork by itself; the worker bootstrap's rlimits do. Apply the
    # implementation's exact limit statements so the probe exercises the real contract.
    limits = python_nodes._bootstrap().split('sys.setrecursionlimit(')[0]
    probe = limits + '''
import json, os, socket
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
    def no_process(*args, **kwargs):
        raise AssertionError('Runtime identity must only read files')
    monkeypatch.setattr(python_nodes.subprocess, 'Popen', no_process)
    identity = python_nodes.runtime_identity()
    assert identity['available'] is True
    assert identity['sandbox_executable']['path'].endswith('bwrap')
    assert identity['stdlib']['files'] > 0 and identity['native_extensions']['files'] > 0
    assert identity['os_build']['kernel_release']
    assert identity['interpreter_flags'] == ['-I', '-S', '-B']
    assert python_nodes.runtime_identity() == identity
