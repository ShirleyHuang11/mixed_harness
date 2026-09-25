"""Sandboxed execution of python nodes.

A python node is a source string that defines `run(obs, nodes, memory)` and helper functions, nothing else.
`validate_source` rejects everything outside a small, explicit subset of Python before the source is ever
sent anywhere: no imports, classes, attribute writes, private names, or unsafe builtins.

Execution happens in a **warm worker**: one sandboxed interpreter per pool slot, started on first use and
reused for later nodes. The sandbox is `bwrap` on Linux (empty root, read-only /usr and the interpreter
prefix, no network, no host environment, tmpfs /tmp) and `sandbox-exec` on macOS. The worker sets memory,
CPU and wall-time limits per job; a job that exceeds them, or breaks the protocol, kills the worker, and the
next job starts a fresh one. Without a supported sandbox there is no fallback: `SandboxUnavailable`.

Why warm: on the cluster this was built on, starting a sandboxed interpreter costs about 0.5 s; a harness
turn with three python nodes spent more time spawning than judging. A warm worker pays that once per pool
slot. Nodes of one runtime share a worker process; they already share a trust domain (one candidate spec).
"""
import ast
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SOURCE_BYTES = 65_536
AST_NODES = 10_000
INPUT_BYTES = 2 * 1024 * 1024
OUTPUT_BYTES = 1024 * 1024
WALL_SECONDS = 5.0
CPU_SECONDS = 2.0
MEMORY_BYTES = 512 * 1024 * 1024
START_SECONDS = 30.0

SAFE_BUILTINS = ('abs all any bool dict enumerate filter float int isinstance len list map max min next range '
                 'reversed round set sorted str sum tuple zip ValueError TypeError KeyError IndexError '
                 'ZeroDivisionError ArithmeticError Exception True False None').split()
MATH_NAMES = ('ceil floor sqrt exp log log10 log2 log1p pow fabs fsum hypot isclose isfinite isinf isnan trunc '
              'sin cos tan atan atan2 degrees radians pi e inf nan').split()
STATISTICS_NAMES = 'mean fmean median median_low median_high mode multimode pstdev pvariance stdev variance'.split()
METHOD_NAMES = set(('get keys values items copy update setdefault pop append extend insert remove index count sort reverse '
                    'lower upper strip lstrip rstrip replace split rsplit join startswith endswith isdigit isalpha isalnum '
                    'isspace find rfind capitalize title removeprefix removesuffix add discard union intersection '
                    'difference issubset issuperset').split())
FORBIDDEN_NAMES = set(('open exec eval compile input print format globals locals vars getattr setattr delattr hasattr dir '
                       'type object super help breakpoint memoryview bytearray bytes exit quit classmethod staticmethod '
                       'property __import__ __builtins__').split())
ALLOWED_AST = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.Expr,
               ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Lambda, ast.ListComp, ast.SetComp,
               ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
               ast.Name, ast.Load, ast.Store, ast.Subscript, ast.Slice, ast.Attribute, ast.Call, ast.keyword,
               ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
               ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Try, ast.ExceptHandler,
               ast.Raise, ast.JoinedStr, ast.FormattedValue, ast.Starred, ast.NamedExpr)


class SandboxUnavailable(RuntimeError):
    """No supported native sandbox on this machine. Infrastructure, never a candidate score."""


class SourceError(ValueError):
    """The node source is outside the allowed subset."""


class NodeError(RuntimeError):
    """The node ran and failed: raised, timed out, exceeded a limit, or returned a non-JSON value."""


def validate_source(source):
    if not isinstance(source, str) or not source.strip():
        raise SourceError('python source must be a nonempty string')
    if len(source.encode('utf-8')) > SOURCE_BYTES:
        raise SourceError(f'python source exceeds {SOURCE_BYTES} bytes')
    try:
        tree = ast.parse(source, mode='exec')
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise SourceError(f'invalid python source: {exc}') from None
    items = list(ast.walk(tree))
    if len(items) > AST_NODES:
        raise SourceError('python source is too large')
    allowed_attrs = METHOD_NAMES | set(MATH_NAMES) | set(STATISTICS_NAMES)
    for item in items:
        if not isinstance(item, ALLOWED_AST):
            raise SourceError(f'disallowed syntax: {type(item).__name__}')
        if isinstance(item, ast.Name) and (item.id.startswith('_') or item.id in FORBIDDEN_NAMES):
            raise SourceError(f'disallowed name: {item.id}')
        if isinstance(item, ast.arg) and (item.arg.startswith('_') or item.annotation is not None):
            raise SourceError('private argument names and annotations are not allowed')
        if isinstance(item, ast.FunctionDef) and (item.name.startswith('_') or item.name in FORBIDDEN_NAMES
                                                   or item.decorator_list or item.returns is not None):
            raise SourceError('decorators, annotations and private function names are not allowed')
        if isinstance(item, ast.Attribute):
            if not isinstance(item.ctx, ast.Load):
                raise SourceError('attribute writes are not allowed')
            if item.attr.startswith('_') or item.attr not in allowed_attrs:
                raise SourceError(f'disallowed attribute: {item.attr}')
        if isinstance(item, ast.comprehension) and item.is_async:
            raise SourceError('async comprehensions are not allowed')
        if isinstance(item, ast.Constant) and not isinstance(item.value, (str, int, float, bool, type(None))):
            raise SourceError('only JSON-compatible literals are allowed')
    for item in tree.body:
        if isinstance(item, ast.FunctionDef):
            continue
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
            continue
        raise SourceError('module scope may only define functions (and docstrings)')
    entry = [item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'run']
    if len(entry) != 1:
        raise SourceError('define exactly one run(obs, nodes, memory)')
    args = entry[0].args
    if ([a.arg for a in args.args] != ['obs', 'nodes', 'memory'] or args.posonlyargs or args.vararg or args.kwarg
            or args.kwonlyargs or args.defaults):
        raise SourceError('the entrypoint must be run(obs, nodes, memory) with no defaults')
    return source


# ---------------------------------------------------------------------------------------------------------
# The program that runs inside the sandbox. Standard library only; reads jobs from stdin until EOF.
# ---------------------------------------------------------------------------------------------------------
WORKER_PROGRAM = r'''
import json, math, resource, signal, statistics, sys, time
CONF = json.loads(sys.stdin.readline())
resource.setrlimit(resource.RLIMIT_AS, (CONF["memory_bytes"], CONF["memory_bytes"]))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
try:
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
except (ValueError, OSError):
    pass
SAFE = {}
import builtins as _b
for name in CONF["safe_builtins"]:
    SAFE[name] = getattr(_b, name)
class Timeout(Exception):
    pass
def _alarm(signum, frame):
    raise Timeout("wall" if signum == signal.SIGALRM else "cpu")
signal.signal(signal.SIGALRM, _alarm)
signal.signal(signal.SIGVTALRM, _alarm)
out = sys.stdout.buffer
inp = sys.stdin.buffer
def _send(obj):
    data = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
    out.write(("%010d\n" % len(data)).encode("ascii")); out.write(data); out.flush()
def _recv():
    head = inp.readline()
    if not head:
        return None
    size = int(head.strip())
    data = b""
    while len(data) < size:
        chunk = inp.read(size - len(data))
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode("utf-8"))
_send({"ready": True})
while True:
    job = _recv()
    if job is None:
        break
    started = time.perf_counter()
    reply = {"ok": False, "error": "unknown"}
    signal.setitimer(signal.ITIMER_REAL, job["wall_seconds"])
    signal.setitimer(signal.ITIMER_VIRTUAL, job["cpu_seconds"])
    try:
        namespace = {"__builtins__": dict(SAFE), "math": math, "statistics": statistics}
        exec(compile(job["source"], "<node>", "exec"), namespace)
        value = namespace["run"](job["obs"], job["nodes"], job["memory"])
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > job["output_bytes"]:
            reply = {"ok": False, "error": "output exceeds %d bytes" % job["output_bytes"]}
        else:
            reply = {"ok": True, "value": json.loads(encoded)}
    except Timeout as exc:
        reply = {"ok": False, "error": "%s time limit exceeded" % exc, "fatal": True}
    except MemoryError:
        reply = {"ok": False, "error": "memory limit exceeded", "fatal": True}
    except RecursionError:
        reply = {"ok": False, "error": "recursion limit exceeded"}
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError, AttributeError, NameError, AssertionError, LookupError, RuntimeError) as exc:
        reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:500])}
    except Exception as exc:
        reply = {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:500])}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
    reply["elapsed_ms"] = (time.perf_counter() - started) * 1000
    try:
        _send(reply)
    except ValueError:
        _send({"ok": False, "error": "output is not JSON", "elapsed_ms": reply["elapsed_ms"]})
    if reply.get("fatal"):
        break
'''


def _linux_command():
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise SandboxUnavailable('python nodes need bwrap (bubblewrap) on Linux')
    base = Path(sys.base_prefix).resolve()
    executable = Path(getattr(sys, '_base_executable', sys.executable)).resolve()
    args = [bwrap, '--unshare-all', '--die-with-parent', '--new-session',
            '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
            '--symlink', 'usr/bin', '/bin']
    if Path('/etc/ld.so.cache').is_file():
        args += ['--ro-bind', '/etc/ld.so.cache', '/etc/ld.so.cache']
    for path in {base, executable.parent}:
        if not str(path).startswith('/usr/'):
            args += ['--ro-bind', str(path), str(path)]
    args += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--chdir', '/tmp', '--']
    return args + [str(executable), '-I', '-S', '-B', '-c', WORKER_PROGRAM]


def _darwin_command():
    if not Path('/usr/bin/sandbox-exec').is_file():
        raise SandboxUnavailable('python nodes need the macOS sandbox-exec')
    base = Path(sys.base_prefix).resolve()
    executable = base / 'Resources/Python.app/Contents/MacOS/Python'
    if not executable.is_file():
        executable = Path(getattr(sys, '_base_executable', sys.executable)).resolve()
    readable = [str(base), '/System/Library', '/usr/lib', '/System/Volumes/Preboot/Cryptexes/OS/System/Library',
                '/System/Cryptexes/OS/System/Library', '/System/Volumes/Preboot/Cryptexes/OS/usr/lib']
    subpaths = ' '.join('(subpath ' + json.dumps(p) + ')' for p in readable)
    profile = '\n'.join(['(version 1)', '(deny default)',
                         '(allow process-exec (literal ' + json.dumps(str(executable)) + '))',
                         '(allow sysctl-read)', '(allow file-read-metadata)',
                         '(allow file-read* file-map-executable ' + subpaths + ')',
                         '(allow file-read* (literal "/dev/urandom") (literal "/dev/null"))',
                         '(allow file-read* file-test-existence (literal "/"))'])
    return ['/usr/bin/sandbox-exec', '-p', profile, str(executable), '-I', '-S', '-B', '-c', WORKER_PROGRAM]


def sandbox_command():
    if sys.platform.startswith('linux'):
        return _linux_command()
    if sys.platform == 'darwin':
        return _darwin_command()
    raise SandboxUnavailable('python nodes need a native sandbox (Linux bwrap or macOS sandbox-exec)')


def sandbox_available():
    try:
        sandbox_command()
        return True
    except SandboxUnavailable:
        return False


class Worker:
    """One sandboxed interpreter. Not thread-safe; the pool serialises access."""

    def __init__(self, command=None):
        self.command = command or sandbox_command()
        self.process = None
        self.jobs = 0
        self.started_ms = None
        self.start_error = None

    def start(self):
        started = time.perf_counter()
        env = {'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONIOENCODING': 'utf-8', 'PYTHONHASHSEED': '0'}
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, env=env, start_new_session=True)
        except OSError as exc:
            raise SandboxUnavailable(f'cannot start the sandbox: {exc}') from None
        conf = {'memory_bytes': MEMORY_BYTES, 'safe_builtins': SAFE_BUILTINS}
        try:
            self.process.stdin.write((json.dumps(conf) + '\n').encode('utf-8'))
            self.process.stdin.flush()
            ready = self._recv(START_SECONDS)
        except (BrokenPipeError, OSError, NodeError) as exc:
            self._kill()
            raise SandboxUnavailable(f'the sandbox worker did not start: {self._stderr() or exc}') from None
        if not (isinstance(ready, dict) and ready.get('ready')):
            self._kill()
            raise SandboxUnavailable(f'the sandbox worker did not report ready: {self._stderr()}')
        self.started_ms = (time.perf_counter() - started) * 1000

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def run(self, source, context, *, wall_seconds=WALL_SECONDS, cpu_seconds=CPU_SECONDS):
        if not self.alive():
            self.start()
        job = {'source': source, 'obs': context['obs'], 'nodes': context['nodes'], 'memory': context['memory'],
               'wall_seconds': wall_seconds, 'cpu_seconds': cpu_seconds, 'output_bytes': OUTPUT_BYTES}
        data = json.dumps(job, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(data) > INPUT_BYTES:
            raise NodeError(f'node input exceeds {INPUT_BYTES} bytes')
        try:
            self.process.stdin.write(('%010d\n' % len(data)).encode('ascii'))
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self._kill()
            raise SandboxUnavailable('the sandbox worker died before accepting a job')
        reply = self._recv(wall_seconds + 2.0)
        self.jobs += 1
        if reply.get('fatal'):
            self._kill()
        if not reply.get('ok'):
            raise NodeError(reply.get('error', 'node failed'))
        return reply['value'], reply.get('elapsed_ms')

    def _recv(self, timeout):
        """Read one length-prefixed JSON reply with a wall timeout; kills the worker on timeout."""
        result = {}

        def reader():
            try:
                head = self.process.stdout.readline()
                if not head:
                    result['error'] = 'worker closed its output'
                    return
                size = int(head.strip())
                data = b''
                while len(data) < size:
                    chunk = self.process.stdout.read(size - len(data))
                    if not chunk:
                        result['error'] = 'worker output truncated'
                        return
                    data += chunk
                result['value'] = json.loads(data.decode('utf-8'))
            except (ValueError, OSError) as exc:
                result['error'] = f'protocol error: {exc}'

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self._kill()
            raise NodeError(f'node exceeded the wall limit ({timeout:.1f}s) and the worker was killed')
        if 'error' in result:
            stderr = self._stderr()
            self._kill()
            raise NodeError(result['error'] + (f': {stderr}' if stderr else ''))
        return result['value']

    def _stderr(self):
        try:
            if self.process and self.process.stderr and self.process.poll() is not None:
                return self.process.stderr.read().decode('utf-8', 'replace').strip()[-500:]
        except (OSError, ValueError):
            pass
        return ''

    def _kill(self):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except (OSError, AttributeError):
                pass

    def close(self):
        self._kill()


class WorkerPool:
    """A bounded set of warm workers shared by the threads of one runtime.

    Workers are started from one long-lived thread. `bwrap --die-with-parent` uses PR_SET_PDEATHSIG, which
    fires when the *thread* that spawned the sandbox exits; a worker started from a short-lived executor
    thread would be killed at the end of every run.
    """

    def __init__(self, size=4):
        self.size = max(1, int(size))
        self._idle = []
        self._count = 0
        self._lock = threading.Lock()
        self._slot = threading.Semaphore(self.size)
        self._requests = queue.Queue()
        self._starter = threading.Thread(target=self._serve_starts, name='mixedharness-sandbox-starter', daemon=True)
        self._starter.start()
        self.stats = {'jobs': 0, 'starts': 0, 'start_ms': 0.0}

    def _serve_starts(self):
        while True:
            item = self._requests.get()
            if item is None:
                return
            worker, done = item
            try:
                worker.start()
                worker.start_error = None
            except Exception as exc:   # reported to the requesting thread
                worker.start_error = exc
            done.set()

    def _start(self, worker):
        done = threading.Event()
        self._requests.put((worker, done))
        if not done.wait(START_SECONDS + 10):
            raise SandboxUnavailable('the sandbox worker did not start in time')
        if worker.start_error is not None:
            raise worker.start_error

    def run(self, source, context, **limits):
        self._slot.acquire()
        worker = None
        try:
            with self._lock:
                worker = self._idle.pop() if self._idle else None
                if worker is None:
                    worker = Worker()
                    self._count += 1
            if not worker.alive():
                self._start(worker)
                with self._lock:
                    self.stats['starts'] += 1
                    self.stats['start_ms'] += worker.started_ms or 0.0
            value, elapsed = worker.run(source, context, **limits)
            with self._lock:
                self.stats['jobs'] += 1
            return value, elapsed
        finally:
            if worker is not None:
                with self._lock:
                    self._idle.append(worker)
            self._slot.release()

    def close(self):
        with self._lock:
            workers, self._idle = self._idle, []
        for worker in workers:
            worker.close()
        self._requests.put(None)


_default_pool = None
_default_lock = threading.Lock()


def default_pool():
    global _default_pool
    with _default_lock:
        if _default_pool is None:
            _default_pool = WorkerPool(size=int(os.environ.get('MIXEDHARNESS_SANDBOX_WORKERS', '4')))
        return _default_pool


def run_python(source, context, pool=None, **limits):
    """Validate and run a python node; returns (value, elapsed_ms_inside_worker)."""
    validate_source(source)
    return (pool or default_pool()).run(source, context, **limits)


def runtime_identity():
    """What the sandboxed interpreter is, for frozen artifacts: interpreter path and version, sandbox binary."""
    identity = {'platform': sys.platform, 'python': sys.version.split()[0],
                'base_prefix': str(Path(sys.base_prefix).resolve()), 'available': sandbox_available()}
    if identity['available']:
        command = sandbox_command()
        identity['sandbox'] = command[0]
        identity['interpreter'] = command[command.index('--') + 1] if '--' in command else command[3]
    return identity
