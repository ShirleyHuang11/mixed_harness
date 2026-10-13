"""The pluggable judge (log-probabilities over option labels) and the turn audit, against a fake logprob server."""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mixedharness import audit
from mixedharness.jev import JevClient
from mixedharness.judge import LogprobJudge, label_logprobs, make_judge, parse_judge_spec, question_prompt, softmax


class FakeServer:
    """Answers every chat completion with log-probabilities that favour the label named in the prompt's first option."""

    def __init__(self, favourite='A', spread=2.0):
        self.favourite, self.spread, self.calls = favourite, spread, []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
                server.calls.append(body)
                top = [{'token': server.favourite, 'logprob': -0.1}, {'token': ' B', 'logprob': -0.1 - server.spread}, {'token': 'C.', 'logprob': -0.1 - 2 * server.spread}]
                payload = {'choices': [{'message': {'role': 'assistant', 'content': server.favourite}, 'logprobs': {'content': [{'token': server.favourite, 'logprob': -0.1, 'top_logprobs': top}]}}],
                           'usage': {'prompt_tokens': 50, 'completion_tokens': 1}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.httpd.server_port}/v1'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def fake():
    s = FakeServer()
    yield s
    s.close()


def test_parse_judge_spec_forms():
    assert parse_judge_spec('jev') == {'kind': 'jev', 'model': 'jev-1.13.0', 'base_url': None}
    assert parse_judge_spec('jev:jev-2')['model'] == 'jev-2'
    assert parse_judge_spec('logprob:Qwen/Qwen3-4B@http://127.0.0.1:8790/v1/') == {'kind': 'logprob', 'model': 'Qwen/Qwen3-4B', 'base_url': 'http://127.0.0.1:8790/v1'}
    assert parse_judge_spec('mock')['kind'] == 'mock'
    with pytest.raises(ValueError):
        parse_judge_spec('logprob:no-url')
    with pytest.raises(ValueError):
        parse_judge_spec('bedrock:x')


def test_make_judge_kinds(fake):
    assert isinstance(make_judge('mock'), JevClient) and make_judge('mock').mock
    assert make_judge('jev').transport == 'typesafe'
    j = make_judge(f'logprob:m@{fake.url}', temperature=0.5)
    assert isinstance(j, LogprobJudge) and j.transport == 'logprob' and j.metadata['temperature'] == 0.5


def test_label_logprobs_tolerates_token_spellings():
    lps = label_logprobs({'A': -0.5, ' B': -1.0, 'C.': -2.0, 'b': -0.2, 'zzz': 0.0}, ['A', 'B', 'C', 'D'])
    assert lps['A'] == -0.5 and lps['B'] == -0.2 and lps['C'] == -2.0 and lps['D'] == -30.0


def test_softmax_temperature_flattens():
    sharp = softmax({'A': 0.0, 'B': -3.0}, 1.0)
    flat = softmax({'A': 0.0, 'B': -3.0}, 3.0)
    assert sharp['A'] > flat['A'] > 0.5 and abs(sum(flat.values()) - 1) < 1e-9


def test_question_prompt_labels_every_option_type():
    text, labels, ids = question_prompt({'k': 1}, {'type': 'choice', 'instructions': 'pick', 'criteria': {'x': 'X', 'y': 'Y'}})
    assert labels == ['A', 'B'] and ids == ['x', 'y'] and 'A. x: X' in text and '"k": 1' in text
    _, labels, ids = question_prompt({}, {'type': 'noul', 'instructions': 'yes?'})
    assert ids == ['yes', 'no']
    _, labels, ids = question_prompt({}, {'type': 'score', 'instructions': 's', 'criteria': ['bad', 'ok', 'good']})
    assert ids == ['0', '1', '2'] and labels == ['A', 'B', 'C']


def test_logprob_judge_answers_all_question_types(fake):
    j = make_judge(f'logprob:m@{fake.url}')
    questions = {'route': {'type': 'choice', 'instructions': 'r', 'criteria': {'closed': 'c', 'escalate': 'e'}},
                 'fits': {'type': 'noul', 'instructions': 'f'},
                 'q': {'type': 'score', 'instructions': 's', 'criteria': ['no', 'yes']}}
    out = j.judge({'conversation': []}, questions)
    a = out['answers']
    assert a['route']['choice'] == 'closed' and a['route']['probabilities']['closed'] > 0.8
    assert abs(sum(a['route']['probabilities'].values()) - 1) < 1e-9
    assert a['fits']['noul'] > 0.8                      # label A = yes
    assert a['q']['score'] < 0.5                        # level 0 favoured
    assert out['cost_usd'] == 0.0 and out['usage']['input_tokens'] == 150 and 'logprobs' in out
    assert j.stats['calls'] == 1 and len(fake.calls) == 3
    body = fake.calls[0]
    assert body['max_tokens'] == 1 and body['logprobs'] is True and body['chat_template_kwargs'] == {'enable_thinking': False}
    # the spec's jev_model is ignored: this judge has one model
    assert j.judge({'conversation': []}, questions, model='jev-1.13.0')['model'] == 'm'


def test_logprob_judge_temperature_changes_confidence(fake):
    hot = make_judge(f'logprob:m@{fake.url}', temperature=4.0).judge({}, {'r': {'type': 'choice', 'instructions': 'r', 'criteria': {'a': 'a', 'b': 'b'}}})
    cold = make_judge(f'logprob:m@{fake.url}', temperature=0.5).judge({}, {'r': {'type': 'choice', 'instructions': 'r', 'criteria': {'a': 'a', 'b': 'b'}}})
    assert cold['answers']['r']['confidence'] > hot['answers']['r']['confidence']


def _trace():
    tools = [{'type': 'function', 'function': {'name': 'get_order', 'description': 'Look up an order by id.', 'parameters': {}}}]
    msgs = [{'role': 'system', 'content': 'be helpful'},
            {'role': 'user', 'content': 'Where is order #1?'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'get_order', 'arguments': '{"order_id": "#1"}'}}], 'usage': {'prompt_tokens': 1000, 'completion_tokens': 20}},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"status": "shipped"}'},
            {'role': 'assistant', 'content': 'It has shipped.', 'usage': {'prompt_tokens': 1200, 'completion_tokens': 10}}]
    return {'id': 'conv-1', 'tools': tools, 'messages': msgs}


def test_audit_messages_format_with_mock_judge(tmp_path):
    path = tmp_path / 'traces.jsonl'
    path.write_text(json.dumps(_trace()) + '\n')
    report, rows = audit.run_audit([str(path)], fmt='messages', judge_spec='mock', price_model='gpt-5.5', out_dir=str(tmp_path / 'out'))
    assert report['conversations'] == 1 and report['turns'] == 2 and report['errors'] == 0
    assert rows[0]['actual'] == 'tool:get_order' and rows[1]['actual'] == 'reply'
    assert rows[0]['priced_by'] == 'usage_x_list_price' and rows[0]['llm_cost_usd'] == pytest.approx((1000 * 5.0 + 20 * 30.0) / 1e6)
    assert [r['tau'] for r in report['curve']['rows']] == list(audit.TAUS)
    assert report['curve']['priced'] and all(0 <= r['cost_ratio'] <= 1.0 + 1e-9 for r in report['curve']['rows'])
    assert (tmp_path / 'out' / 'report.md').read_text().startswith('# Turn audit')
    assert 'get_order' in report['tools']


def test_audit_tau2_format_reads_cost_from_trace(tmp_path):
    sim = {'task_id': '7', 'reward_info': {'reward': 1.0}, 'messages': [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'Hi! How can I help you today?', 'cost': 0.0},
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'Hello, how can I help?', 'cost': 0.02, 'usage': {'prompt_tokens': 4000, 'completion_tokens': 10}},
        {'role': 'user', 'content': 'cancel #W1'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'x', 'name': 'cancel_pending_order', 'arguments': {'order_id': '#W1'}}], 'cost': 0.03},
        {'id': 'x', 'role': 'tool', 'content': 'ok'}]}
    path = tmp_path / 'results.json'
    path.write_text(json.dumps({'simulations': [sim]}))
    report, rows = audit.run_audit([str(path)], fmt='tau2', judge_spec='mock')
    assert report['turns'] == 3 and rows[2]['actual'] == 'tool:cancel_pending_order' and rows[2]['priced_by'] == 'trace'
    assert rows[0]['llm_cost_usd'] == 0.0 and rows[0]['priced_by'] == 'trace'   # a canned greeting costs nothing, and that is a price
    assert report['curve']['priced'] and report['curve']['llm_cost_total_usd'] == pytest.approx(0.05)
    assert rows[2]['reward'] == 1.0 and 'cancel_pending_order' in report['tools']


def test_audit_curve_counts_taken_turns_and_cost():
    rows = [{'route': 'closed', 'pick_p': 0.995, 'agree': True, 'actual': 'tool:a', 'llm_cost_usd': 0.5, 'judge_cost_usd': 0.001},
            {'route': 'closed', 'pick_p': 0.7, 'agree': False, 'actual': 'reply', 'llm_cost_usd': 0.5, 'judge_cost_usd': 0.001},
            {'route': 'escalate', 'pick_p': 0.999, 'agree': True, 'actual': 'reply', 'llm_cost_usd': 1.0, 'judge_cost_usd': 0.001}]
    c = audit.curve(rows, taus=(0.6, 0.99))
    by = {r['tau']: r for r in c['rows']}
    assert by[0.6]['turns_taken'] == 2 and by[0.6]['agreement'] == 0.5 and by[0.6]['cost_ratio'] == pytest.approx((1.0 + 0.003) / 2.0)
    assert by[0.99]['turns_taken'] == 1 and by[0.99]['agreement'] == 1.0 and by[0.99]['tool_turns_taken'] == 1


def test_example_traces_audit_with_the_mock_judge():
    from pathlib import Path
    report, rows = audit.run_audit([str(Path(__file__).resolve().parents[1] / 'examples/audit/traces.jsonl')], fmt='messages', judge_spec='mock', price_model='gpt-5.5')
    assert report['conversations'] == 2 and report['turns'] == 10 and report['errors'] == 0 and report['curve']['priced']
    assert {r['actual'] for r in rows} == {'tool:find_user_id_by_email', 'tool:get_order_details', 'tool:cancel_pending_order', 'reply'}


def test_logprob_judge_rereads_cache_at_its_temperature(tmp_path):
    from mixedharness.judge import make_judge
    s = FakeServer()
    try:
        spec = f'logprob:fake@{s.url}'
        state = {'x': 1}
        questions = {'q': {'type': 'choice', 'instructions': 'pick', 'criteria': {'a': 'A', 'b': 'B', 'c': 'C'}}}
        sharp = make_judge(spec, cache_dir=str(tmp_path), cache_namespace='t', temperature=1.0).judge(state, questions)
        flat = make_judge(spec, cache_dir=str(tmp_path), cache_namespace='t', temperature=5.0).judge(state, questions)
    finally:
        s.close()
    assert not sharp['cache_hit'] and flat['cache_hit']
    p_sharp, p_flat = sharp['answers']['q']['probabilities'], flat['answers']['q']['probabilities']
    assert max(p_sharp.values()) > max(p_flat.values())
    assert abs(sum(p_flat.values()) - 1) < 1e-6
