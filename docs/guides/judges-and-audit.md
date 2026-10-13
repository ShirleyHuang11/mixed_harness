# Choosing a judge, and measuring your own agent first

Two things make Mixed Harness usable outside one vendor's customers: the judge behind every `jev` node is a
configuration string, and the question "how much of my agent's work is closed choices?" can be answered on a
recorded trace before anything is built. This guide covers both.

## The judge is a configuration string

Every node of kind `jev` calls `judge(state, questions)` and reads typed answers with probabilities. Which model
answers is set with `--judge` on every script, or with `make_judge(spec)` in code:

| spec | what answers | needs |
|---|---|---|
| `jev` (default) | Jev, TypeSafe's judgment model, `jev-1.13.0` | `TYPESAFE_API_KEY` |
| `jev:<model>` | another Jev model id | `TYPESAFE_API_KEY` |
| `logprob:<model>@<base_url>` | any model served on an OpenAI-compatible endpoint that returns first-token `top_logprobs` | the server; `JUDGE_API_KEY` only if it wants one |
| `mock` | deterministic pseudo-answers | nothing (plumbing tests only) |

The log-probability judge asks one question per request, with the state as JSON, the instructions, and the
options labelled A, B, C ... It reads the probability the model puts on each label as its next token and reports
that distribution after a temperature. Servers that return the needed field: vLLM (`--max-logprobs 20`),
llama.cpp's server, SGLang, LM Studio, and the bundled `scripts/judge_server.py` over a plain transformers model:

```bash
uv pip install torch transformers
python scripts/judge_server.py --model Qwen/Qwen3-4B --port 8790        # GPU if present, else CPU
python scripts/tau2_compare.py --arms routed --judge logprob:Qwen/Qwen3-4B@http://127.0.0.1:8790/v1 ...
```

Qwen3 chat templates get `enable_thinking: false` so the first token is the answer; other chat models work as
they are. The temperature is a view over the judge's cached log-probabilities: a cached answer is re-read at
the temperature the judge was made with, so a fitted temperature re-renders a replay or an audit without new
requests (`--judge-temperature` on `mixedharness audit`; the policy file carries it for the runs).

## A judge is only as good as its probabilities

The harness hands a turn to the judge when its probability on the pick clears the threshold for that state type.
So a judge is usable when its probability means what it says, and the threshold table has to be fitted for the
judge that will run. The workflow, start to finish, for a new judge:

```bash
# 1. replay the recorded LLM-only turns on the new judge (the same questions the harness asks)
python scripts/stage0_label_replay.py --results runs/tau2/<run>/llm_only/results.json \
    --judge logprob:Qwen/Qwen3-4B@http://127.0.0.1:8790/v1 --out runs/judges/qwen3-4b/stage0
# 2. the reliability table, a fitted temperature, the kill test, and a calibrated replay
python scripts/calibrate_judge.py --replay runs/judges/qwen3-4b/stage0/<stamp>/replay.jsonl --out runs/judges/qwen3-4b/judge.json
# 3. the threshold table for this judge; the policy file records which judge and temperature it is for
python scripts/fit_policy.py --replay runs/judges/qwen3-4b/stage0/<stamp>/replay.calibrated.jsonl --tau-only \
    --judge-json runs/judges/qwen3-4b/judge.json --out runs/judges/qwen3-4b/policy.json
# 4. run with it; --judge can be omitted when the policy names the judge
python scripts/tau2_compare.py --arms routed --policy runs/judges/qwen3-4b/policy.json --judge logprob:Qwen/Qwen3-4B@http://127.0.0.1:8790/v1
```

The kill test in step 2 is fixed: after the temperature fit, the bin of turns where the judge reports a
probability of at least 0.99 must agree with the LLM on at least 85% of them and hold at least 10% of closed
turns. A judge that fails it has no bin a threshold can trust, and the right next step is a larger model, not
a longer prompt. Passing it is the floor: how many turns a judge takes is decided by the per-cluster table
fitted in step 3.

## Judges measured so far

The same 1,389 recorded LLM-only retail turns for every judge (the stage-0 replay of run
`20261001-stage2-v3a-merged`; Jev's replay is `runs/stage0/20261006-131412`, the open models' are under
`runs/judges/<model>/stage0`). The last column is the audit of the 114 gpt-5.5 conversations with that judge.

| judge | agreement with the LLM on closed turns | >= 0.99 bin: share of closed turns / agreement | closed turns its table hands over at >= 90% agreement | audit at tau 0.90: share / agreement / cost |
|---|---|---|---|---|
| Jev `jev-1.13.0`, as reported | 72% (1,002 turns) | 77% / 74% | 57% (482 of 853, clusters of 10+ turns) | 56% / 95% / 0.50x |
| Qwen3-4B, log-probabilities, T = 5.0 | 52% (982 turns) | 20% / 99% | 30% (247 of 835) | 47% / 58% / 0.57x |

Jev's reported probability is not calibrated on this replay (its top bin holds 77% of turns and is right on 74%
of them) and cannot be re-tempered; its per-cluster table is what makes it usable, and that table hands over
nearly twice the turns of the 4B judge. The 4B judge is calibrated in its top bin after the fit, and the bin is
small: outside the pre-authentication clusters no threshold reaches 90% agreement, so its table leaves those
turns to the LLM. On the audit's open 17-way choice its agreement barely moves with the threshold (47% at 0.60,
58% at 0.90) where Jev's climbs to 99%.

## The ruler: `mixedharness audit`

Before building anything, measure what there is to gain. The audit takes a recorded trace of any tool-using
agent and asks the judge, on every assistant turn, which of the agent's tools (or a reply) comes next, with the
agent's recorded action as the answer key. No compiler, no harness, no benchmark.

```bash
mixedharness audit runs/tau2/<run>/llm_only/results.json --format tau2 --judge jev
mixedharness audit traces.jsonl --format messages --judge logprob:Qwen/Qwen3-4B@http://127.0.0.1:8790/v1 --judge-temperature 5.0 --price-model gpt-5.5
```

Formats: `tau2` is a tau2-bench `results.json`; `messages` is JSON lines of OpenAI-style conversations,
`{"id", "tools": [tool schemas], "messages": [{"role", "content", "tool_calls", "usage"?, "cost"?}]}`
(`examples/audit/traces.jsonl` holds two). Tool
descriptions come from the trace's tool schemas, or from `--tools tools.json` (`{name: description}`); the
agent's policy or system prompt can be shown to the judge with `--policy-text`. Per-turn LLM cost comes from
the trace's own `cost`, or from its token usage at `--price-model`'s list price.

The report is a curve over the threshold:

| tau | turns the judge takes | pick = agent action | cost of the trace |
|---|---|---|---|
| 0.90 | 777 (56%) | 95% | 0.50x |
| 0.99 | 402 (29%) | 99% | 0.79x |

(Jev on 114 retail conversations of a gpt-5.5 agent, 1,388 turns, 2026-10-13.) Read it as: at a threshold of
0.90 this judge would take 56% of the turns and pick what the agent picked on 95% of them, and the trace would
have cost half. The share is what a harness built on this judge could move off the LLM; the agreement is the
accuracy it would pay on those turns. One caveat: naming the right tool is not yet filling its arguments, which
the harness's compiler does from state, so the audit bounds what a harness can take rather than measuring one.
