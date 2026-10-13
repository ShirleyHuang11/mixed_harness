# The pluggable judge and the turn audit

Written 2026-10-13, before any open-model judge had answered a question. Two things stood between the harness
and anyone outside TypeSafe's customers: the judge was Jev only (one key almost nobody has), and the only way
to see whether the idea applies to your own agent was to port the whole harness to it. This note fixes both and
records the predictions first.

## Three paths, ranked by upside

1. **A ruler anyone can run on their own traces** (`mixedharness audit`). If it works, a team with any recorded
   agent learns in minutes what share of its turns are closed choices and what a judge would save, without
   adopting anything. Startable today: the judge, the trace formats and the cost fields all exist.
2. **A judge behind any model that returns log-probabilities** (`--judge logprob:<model>@<base_url>`). If it
   works, the requirements shrink to one LLM key, or none with a local model; the harness becomes installable by
   everyone. Startable today: the judge interface is one method; vLLM, llama.cpp and a 100-line transformers
   server all expose first-token log-probabilities.
3. A second domain (airline) with an LLM-written option catalog. Highest upside for the claim "general", but it
   needs a new compiler first, so it is the upgrade condition for the two above.

Both 1 and 2 are built here; 3 waits.

## What is built

- `mixedharness/judge.py`: `make_judge(spec)` returns Jev, a mock, or `LogprobJudge`. The log-probability judge
  turns each question into a multiple-choice prompt (state as JSON, instructions, options labelled A, B, C ...),
  asks for one token with `logprobs`, and reports the softmax over the option labels, after a temperature. The
  rest of the harness does not know which judge it has: `jev` nodes call `judge(state, questions)` as before.
- `scripts/judge_server.py`: an OpenAI-compatible chat endpoint over a transformers model that returns the
  first-token top log-probabilities. Zero keys. vLLM or llama.cpp can stand in for throughput.
- `scripts/calibrate_judge.py`: the reliability table of a judge from a stage-0 replay, a fitted temperature on
  the raw label log-probabilities, the kill test, and a calibrated replay for `fit_policy.py`.
- `--judge` on `stage0_label_replay.py`, `tau2_compare.py`, `long_run.py`, `live_run.py`; the policy file
  carries `judge: {spec, temperature}` so a table is tied to the judge it was fitted for.
- `mixedharness/audit.py` and `mixedharness audit`: the ruler. Input a tau2 `results.json` or OpenAI-style
  conversation JSON lines; the judge answers route and action on every assistant turn with the recorded action
  as the key; the report is a curve over tau: share taken, agreement on those turns, and the trace re-priced.

## Predictions, written before the runs

Positive control: Jev on the audit of the 114 LLM-only retail conversations (gpt-5.5 agent, 2026-10-06) must
match the agent's recorded action on >= 90% of the turns it would take at tau = 0.99. Below that the audit's
generic questions are broken and nothing else in this note is read.

| quantity | prediction | kill |
|---|---|---|
| audit with Jev, share taken at tau 0.99 | 35-50% of turns | < 20% |
| audit with Jev, agreement on taken turns at 0.99 | >= 90% | < 85% |
| audit with Jev, cost of the trace at 0.99 | 0.55-0.70x | > 0.85x |
| Qwen3-4B logprob judge, stage-0 replay, >= 0.99 bin after calibration | agreement 85-92%, share 20-40% of closed turns | agreement < 85% or share < 10% |
| Qwen3-4B judge overall agreement on closed turns | 70-80% (Jev: 85%) | < 60% |
| routed run with the Qwen3-4B judge and its own table, 114 tasks | pass^1 0.74-0.80, cost 0.78-0.88x of llm_only | pass^1 < 0.70 |

Collapse factor to look for: the >= 0.99 bin of a calibrated 4B judge should agree at 2x the rate of its < 0.60
bin; if the bins are flat the probability carries no information and no threshold can use it.

Decision table: if the 4B judge passes its kill test, the README's requirements become "a TypeSafe key, or any
model served with log-probabilities", with the two judges' numbers side by side. If it fails, the README keeps
Jev as the only judge and says so in one sentence, and the next step is a larger open model (8B / 14B) on the
same replay rather than more tuning of the 4B prompt.

## Results

(filled in by the runs; see below)

### 2026-10-13, the audit with Jev (positive control and first ruler numbers)

114 retail conversations of the gpt-5.5 LLM-only agent (run 20261006-235933), 1,388 assistant turns, 16 tools,
Jev `jev-1.13.0`. LLM cost of the trace as recorded: $40.05; Jev over every turn: $0.18 (plain) / $0.21 (with
policy).

| tau | plain: share taken / agreement / cost | with the retail policy shown: share / agreement / cost |
|---|---|---|
| 0.60 | 86% / 87% / 0.16x | 84% / 89% / 0.20x |
| 0.80 | 68% / 93% / 0.37x | 63% / 96% / 0.44x |
| 0.90 | 56% / 95% / 0.50x | 43% / 97% / 0.65x |
| 0.95 | 44% / 97% / 0.63x | 22% / 97% / 0.80x |
| 0.99 | 29% / 99% / 0.79x | 12% / 99% / 0.90x |

- 🎯 Positive control passed: 99% agreement on the turns taken at 0.99 (predicted >= 90%).
- ⚠️ Share at 0.99 (29%) came in under the predicted 35-50% and the cost there (0.79x) above the predicted
  0.55-0.70x; neither hit its kill line (20%, 0.85x). At 0.90 the ruler reads 56% / 95% / 0.50x, which is the
  threshold the harness itself runs most state types at.
- The route question carries no signal in the audit (closed on 99-100% of turns): with every tool listed and no
  compiler to say which arguments are known, Jev has no reason to say escalate. The action probability does all the
  work; the report says so when it happens.
- Showing the policy makes Jev more cautious (12% vs 29% at 0.99) without making it more accurate on what it
  takes. The plain audit is the headline; the policy run is the conservative reading.
- Agreement on every turn, no threshold: 82%. The harness's recorded Jev share (42%) sits between the 0.90 and
  0.95 rows, consistent with its per-cluster thresholds.

### 2026-10-13, the Qwen3-4B log-probability judge (stage-0 replay, calibration, audit)

Same 1,389 recorded LLM-only turns as the Jev replay (run 20261001-stage2-v3a-merged), 982 closed; one A100,
the bundled transformers server, 64 minutes for the replay, calibration and both audits (`runs/judges/Qwen-Qwen3-4B/`).

| | as reported (T = 1) | after calibration (T = 5.0) |
|---|---|---|
| >= 0.99 bin: share of closed turns / agreement with the LLM | 88% / 55% | 20% / 99% |
| 0.90-0.99 | 6% / 31% | 46% / 43% |
| 0.60-0.90 | 5% / 26% | 20% / 42% |
| < 0.60 | 2% / 35% | 14% / 26% |
| ECE | 0.46 | 0.34 |

- 🎯 Kill test passed, on its lines: the calibrated >= 0.99 bin agrees with the LLM on 99% of its turns and holds
  20% of closed turns (kill: < 85%, < 10%). The collapse from the top bin to the bottom one is 3.8x (99% vs 26%;
  predicted 2x), so the probability carries information.
- ❌ Overall agreement on closed turns is 52% (predicted 70-80%, kill < 60%). Raw, the model puts >= 0.99 on
  88% of turns and is right on 55% of them; the fitted temperature sits at the top of the grid (5.0), which is the
  fit asking to flatten further. The 4B judge is usable only through its calibrated top bin.
- 📊 The threshold table fitted on the calibrated replay (`policy.json`, tau only) accepts 247 of 982 closed turns
  (25%), all in the pre-authentication clusters (`anon|user|*`), at 0.91-1.00 agreement; every `auth|*` cluster
  gets tau = 1.01 (never). Jev's table on the same replay hands over 42% of turns across the clusters.
- ⚠️ The audit with this judge (temperature 5.0 borrowed from the stage-0 fit) reads 47% / 58% / 0.57x at tau
  0.90 and takes no turn at 0.99 (plain; Jev: 56% / 95% / 0.50x). On the audit's 17-way choice the 4B judge's
  agreement is flat across tau (47% at 0.60, 58% at 0.90): on open choices over every tool its probability
  separates little. The audit's generic question is harder than the harness's clustered one, and a temperature
  fitted on 2-6-way questions caps the reachable probability on 17-way ones; a per-question-size temperature is the
  obvious fix and is not done here.

Decision, per the table above: the kill test passed, so the routed arm runs on all 114 tasks with this judge and
its own table (prediction: pass^1 0.74-0.80, cost 0.78-0.88x; kill pass^1 < 0.70). Qwen3-8B and Qwen3-1.7B run
the same replay for the ladder.
