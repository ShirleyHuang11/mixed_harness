"""A flat JSON run store.

    runs/<run_id>/run.json                    config, status, result, provider metadata
    runs/<run_id>/events.jsonl                one event per line, in order
    runs/<run_id>/candidates/<hash>.json      every spec that was evaluated, with origin and parents
    runs/<run_id>/evaluations.jsonl           one row per (candidate, episode, split): score, usage, error
    runs/<run_id>/traces/<split>/<hash>/<episode>.json   the complete evaluator result, when captured
    runs/<run_id>/reflections/<n>.prompt.json / .response.txt

Everything is written atomically (temp file + rename) so a killed process never leaves a half file.
The store is the only thing `table.md`-style reports read from; nothing is summarised from memory.
"""
import datetime as dt
import json
import os
import re
import threading
import uuid
from pathlib import Path


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp.' + uuid.uuid4().hex[:8])
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, path)


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp.' + uuid.uuid4().hex[:8])
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _safe(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name))[:120]


class RunStore:
    def __init__(self, root='runs'):
        self.root = Path(root)
        self._lock = threading.Lock()

    def run_dir(self, run_id):
        return self.root / _safe(run_id)

    def create_run(self, name, config):
        run_id = dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + _safe(name) + '-' + uuid.uuid4().hex[:6]
        atomic_write_json(self.run_dir(run_id) / 'run.json',
                          {'run_id': run_id, 'name': name, 'created': now_iso(), 'status': 'created', 'config': config})
        self.event(run_id, 'run_created', {'name': name})
        return run_id

    def read_run(self, run_id):
        return read_json(self.run_dir(run_id) / 'run.json', {})

    def update_run(self, run_id, **fields):
        with self._lock:
            run = self.read_run(run_id)
            run.update(fields, updated=now_iso())
            atomic_write_json(self.run_dir(run_id) / 'run.json', run)
        return run

    def event(self, run_id, kind, payload=None):
        with self._lock:
            append_jsonl(self.run_dir(run_id) / 'events.jsonl', {'time': now_iso(), 'event': kind, **(payload or {})})

    def events(self, run_id, kind=None):
        rows = read_jsonl(self.run_dir(run_id) / 'events.jsonl')
        return [r for r in rows if kind is None or r.get('event') == kind]

    def save_candidate(self, run_id, spec, spec_hash, *, origin, parents=(), extra=None):
        path = self.run_dir(run_id) / 'candidates' / (spec_hash + '.json')
        if path.is_file():
            existing = read_json(path)
            if extra:
                existing.update(extra)
                atomic_write_json(path, existing)
            return existing
        record = {'hash': spec_hash, 'origin': origin, 'parents': list(parents), 'saved': now_iso(), 'spec': spec, **(extra or {})}
        atomic_write_json(path, record)
        return record

    def candidates(self, run_id):
        folder = self.run_dir(run_id) / 'candidates'
        return {p.stem: read_json(p) for p in sorted(folder.glob('*.json'))} if folder.is_dir() else {}

    def record_evaluation(self, run_id, row):
        with self._lock:
            append_jsonl(self.run_dir(run_id) / 'evaluations.jsonl', row)

    def evaluations(self, run_id, *, candidate=None, split=None):
        rows = read_jsonl(self.run_dir(run_id) / 'evaluations.jsonl')
        return [r for r in rows if (candidate is None or r.get('candidate') == candidate) and (split is None or r.get('split') == split)]

    def save_trace(self, run_id, split, spec_hash, episode_id, result):
        atomic_write_json(self.run_dir(run_id) / 'traces' / _safe(split) / spec_hash / (_safe(episode_id) + '.json'), result)

    def read_trace(self, run_id, split, spec_hash, episode_id):
        return read_json(self.run_dir(run_id) / 'traces' / _safe(split) / spec_hash / (_safe(episode_id) + '.json'))

    def save_reflection(self, run_id, index, prompt_payload, response_text=None):
        folder = self.run_dir(run_id) / 'reflections'
        atomic_write_json(folder / f'{index:03d}.prompt.json', prompt_payload)
        if response_text is not None:
            atomic_write_text(folder / f'{index:03d}.response.txt', response_text)
        return folder / f'{index:03d}.prompt.json'

    def list_runs(self):
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / 'run.json').is_file())
