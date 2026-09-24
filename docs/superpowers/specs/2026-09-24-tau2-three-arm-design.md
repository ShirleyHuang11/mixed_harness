# τ²-bench three-arm test: routed harness vs Jev-only vs LLM-only

Date: 2026-09-24. Status: design, implementation in `examples/tau2/`, runner in `scripts/tau2_compare.py`.

## Path choice

Three candidate paths, ranked by ceiling. Each line: what changes if it holds, and whether it can start today.

1. **τ²-bench retail (chosen).** Same environment, same user simulator, same tasks and seeds for
   three agents that differ only in who decides and who writes. If it holds, customer-service
   agents get built as code + a judgment model with an LLM only at the free-text edges, and the
   per-turn cost of a tool-using agent drops by an order of magnitude at matched pass^1.
   Startable today: the LLM-only arm runs on the keys we have; the two Jev arms run the moment
   the TypeSafe key stops returning 401.
2. τ³-bench `banking_knowledge` (unsaturated, top 51.3%). Same claim on a harder domain with
   document retrieval. Not startable today: needs the retrieval extra and a retrieval config.
   Upgrade condition: retail arms complete and the harness lands within 10 points of LLM-only.
3. WebArena in accessibility-tree mode (text-only agents at 46–62%). Highest fame. Not startable
   today: needs the Dockerised web environments. Upgrade condition: same as 2, plus a machine
   that can host the containers.

Reason for 1 in one sentence: it is the only path where all three arms can be scored by the same
DB-state ground truth this week.

## Preemption check (ten minutes, non-blocking)

- τ²-bench, arXiv:2506.07982: LLM agents only; no judgment model, no code-first agent.
- TypeSafe workflow evals (evals.typesafe.ai): Jev "workflow" vs LLM "prompt" on four private
  tasks; labels are the consensus of two LLMs. Ours is a public benchmark with DB-state truth.
- Select-then-Solve, arXiv:2604.06753; Meta-Harness, arXiv:2603.28052: route or rewrite whole
  reasoning strategies; neither splits a single turn into a typed judgment and a generation.
- Leaderboard note: 2026 frontier models with tools sit at 98–99% on retail and telecom
  (artificialanalysis.ai/evaluations/tau2-bench). The claim here is therefore **not** SOTA
  accuracy. It is matched accuracy against the same LLM used alone, at a fraction of the cost
  and latency per turn: an oracle-vs-practice pair where both curves are ours to measure.

## Domain

Retail, `base` split, 114 tasks, 16 tools, reward = DB state + natural-language assertions
(112 tasks) or DB state alone (2). Mean 4.8 expected actions per task. Airline (50 tasks, 14
tools) is the replication domain. Telecom is excluded: its reward is on the *user's* device
state, so the agent's turns are mostly free-text instructions and the settled part is thin.

Official user simulator: `gpt-4.1-2025-04-14`, temperature 0, unchanged across arms. NL
assertions are judged by the same model. Both run on the OpenAI key already in `.env`.

## Arms

| Arm | Action type decided by | Tool arguments filled by | User-facing text written by | LLM calls / turn | Jev calls / turn |
|---|---|---|---|---|---|
| LLM-only | tau2 stock `llm_agent`, `claude-sonnet-5` | same LLM | same LLM | 1 | 0 |
| Jev-only | Jev choice over the legal-action catalog | code, from token parsing and Jev choices over values fetched by earlier tools | fixed templates filled by code | 0 | 1–2 |
| Harness | Jev choice over the same catalog | code as above; an llm node only when code cannot fill a field | templates when one applies; llm node when Jev picks `llm_reply` | ≤ 0.3 expected | 1–2 |

All three see the same policy text and the same tool schemas. The Jev-only arm exists to show
where generation is genuinely needed; it is expected to lose. The harness arm is the seed
harness first, then the same seed after GEPA evolution on the train split.

Positive control: the LLM-only arm must reproduce a Claude-class retail pass^1. Threshold 0.60.
Below it the user simulator or environment is broken and nothing else in the batch is read.

## Runtime change

One shared-harness change, documented in `docs/routed-harness.md`: an `llm` node whose
`prompt_expression` evaluates to `null` is skipped and outputs `null`. Code decides whether the
model is needed. Without this every turn would pay for an LLM call whether or not one was used.
Smoke: rerun the classify example under mock providers before and after and diff the artifacts.

## Findings while building (2026-09-24, before any real-Jev run)

- **Sandboxed Python costs 0.5 s per node on this cluster**, independent of the node's work (trivial node
  640 ms median; `python -I -S` alone 16 ms). With four python nodes a turn spent 2.7 s in process spawn.
  Two changes: the graph now asks every question in one Jev request (speculative fan-out: action choice plus
  the selections the action might need), which cut the graph to two python nodes for jev_only and three for
  harness; and the ledger reports model-only latency and sandbox time separately. The latency prediction in
  the table below is on the model-only number. A warm sandbox worker is the proper fix and is an upgrade
  condition, not part of this test.
- **The OpenAI account has no credit**, so tau2's official user simulator (`gpt-4.1-2025-04-14`) cannot run.
  The user simulator and the NL-assertion judge default to `anthropic/claude-sonnet-4-5-20250929`, the same
  for every arm; the manifest records both the model used and tau2's default. The positive-control threshold
  stays at 0.60 but the published gpt-4.1-simulator numbers are not directly comparable.
- Under mock Jev the seed authenticates, loads the profile, the order and its products in the first nine
  turns without an LLM; the mock's constant answers then stall on item selection and the loop breaker
  transfers the call. The real-Jev run is the first reading of whether item selection works.

## Measurement

- Accuracy: pass^1 on the 114-task base split; pass^k with 3 trials if the budget allows.
- Cost: agent LLM cost from tau2's ledger plus Jev cost from response usage ($0.042 per million
  input tokens, docs.typesafe.ai/models). User-simulator cost reported separately, never added.
- Latency: per-turn agent generation time, median and p95, from the same ledger for every arm.
- Instrument counters per arm: `llm_calls`, `jev_calls`, `python_node_runs`, and the actual model
  ids echoed by each provider. Printed into the manifest for every run.

Resolution, one line of arithmetic: 114 tasks at p ≈ 0.8 gives a 95% interval of ±7.4 points;
the 40-task test split gives ±12. Accuracy differences under 8 points are reported as "within
noise". The discriminating axes are cost and latency, where run-to-run noise is under 1.3× and
the expected gap is 5× or more, so effect / noise ≥ 3 holds there.

## Predictions, written before any run

| Quantity | Prediction |
|---|---|
| LLM-only pass^1, retail base | 0.75 ± 0.08 (positive control ≥ 0.60) |
| Jev-only pass^1 | 0.25–0.45 |
| Harness seed pass^1 | ≥ 0.60 and within 10 points of LLM-only |
| Harness seed LLM calls per turn | ≤ 0.3 |
| Harness per-turn median latency vs LLM-only | ≤ 1/5 |
| Harness agent cost per task vs LLM-only | ≤ 1/10 |
| Harness after 5 GEPA rounds on train, scored on test | +5 to +10 points over the seed |
| Null-model collapse: harness vs Jev-only pass^1 | ≥ 1.5× |

If the harness scores within noise of Jev-only, the llm node is not firing; check the counters
before interpreting anything.

## Kill test

If the evolved harness is more than 15 points below LLM-only on the base split, the claim
"cheap decisions match the LLM" is dead on τ². The instrument survives: the tau2 adapter, the
per-turn cost and latency ledger, and the three-arm runner move unchanged to the settled-route
line (ALFWorld / BALROG) where no generation is needed.

## Decision table (also in the runner's header)

| Observation after the smoke batch | Next action, automatic |
|---|---|
| LLM-only < 0.60 | stop; inspect user simulator and environment |
| Jev-only reports llm_calls > 0, or LLM-only reports jev_calls > 0 | stop; instrument broken |
| Harness within 10 points of LLM-only and ≥ 3× cheaper | run 5 GEPA rounds on train (74), score test (40); then 3 trials on base for pass^k |
| Harness 10–15 points below LLM-only | evolution first (it is the designed corrective), then re-score |
| Harness > 15 points below after evolution | kill; keep instrument; move to settled line |

## Budget

Rough: 15 turns per task, ~12k input tokens per LLM turn. LLM-only ≈ $0.5 per task on
`claude-sonnet-5`; user simulator ≈ $0.3 per task on `gpt-4.1`. Three arms on 114 tasks ≈ $150
plus evolution (74 train tasks × 5 rounds × 2 candidates ≈ $400). Realised cost is written to
the manifest; nothing is estimated after the fact.

## Files

- `examples/tau2/task.json` — TaskDescription: catalog action space, environment evaluator.
- `examples/tau2/adapter.py` — `load(split)`, `evaluate_episode(spec, episode, jev, llm)`.
- `examples/tau2/agent.py` — `PipelineAgent`, a tau2 `HalfDuplexAgent` that runs one
  `PipelineRuntime` call per turn and carries memory across turns.
- `examples/tau2/seeds.py` — the Jev-only and harness seed specs.
- `scripts/tau2_compare.py` — runs the arms, writes `runs/tau2/<stamp>/manifest.json`,
  per-arm results, and `table.md` generated from the artifacts.
- `tests/test_tau2.py` — end-to-end under mock providers on tau2's `mock` domain.
