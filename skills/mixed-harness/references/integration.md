# Integration reference

## The spec and the runtime

A harness is a JSON graph (`mixedharness/spec.py`): `python` nodes (sandboxed source defining
`run(obs, nodes, memory)`), `jev` nodes (state + questions), `llm` and `agent` nodes, `expression` nodes.
`Runtime(spec, jev, llm).run(obs, memory)` executes one turn and returns `output`, `memory`, `nodes` (every
node's output) and a `trace`. `docs/guides/routed-harness.md` is the full reference.

## The routed seed (`examples/tau2/seeds.py`)

Nodes, in order: `compile` (python) → `judge` (jev) → `route` (python) → `decision` (python) / `agent` (agent)
→ `guard` (python). The optional menu path (`menu`, `menu_q`, `menu_pick`) is off by default.

- `compile` returns `{facts, memory, options, needs, closed, cluster, questions, candidates, state}`.
  `options` maps option ids (`tool:<name>[:<entity>]`, `reply:<template>`) to one-line descriptions Jev reads;
  `needs` maps option ids to the arguments code cannot fill; `closed` is true when the catalog is non-empty,
  `needs` is empty, the request is covered and no request remains with only replies listed; `cluster` is the
  state type; `questions` is the Jev request; `state` is what Jev sees (policy, last turns, facts, focused
  record, options, needs).
- `route` reads the policy embedded in the seed (`policy["tau"]` per cluster, `NEVER` = 1.01) and returns the
  path and its source (`jev`, `low_confidence`, `code_override`, `trajectory`, ...).
- `guard` returns `{action, memory, chosen, guard, routing}`; `routing` is the record the step table reads.

`routed_seed(llm_model, policy=...)` embeds a policy JSON (`scripts/fit_policy.py` output) into the seed, so
the table hashes with the harness version.

## Scripts

| Script | Does |
|---|---|
| `scripts/stage0_label_replay.py` | labels recorded LLM-only turns closed/open, scores the compiler flag, replays Jev; writes `runs/stage0/<stamp>/{labels,replay}.jsonl` |
| `scripts/fit_policy.py` | `--tau-only --replay ... [--ledger-replay ...]` thresholds; without `--tau-only`, also code rows (`--steps`) and split rows (`--ledgers`) |
| `scripts/tau2_compare.py` | the arms on τ²-bench; `--arms`, `--policy`, `--seed-module`, `--llm-backend codex --proxy-base`, `--task-ids`, `--seed` |
| `scripts/tau2_merge.py` | merge partial runs by task id |
| `scripts/long_run.py` | stream + held-out curve; `--tau-only`, `--seed-module`, `--first-pass`, `--seed-stream-dirs` to resume |
| `scripts/learn_splits.py` | split rows from ledgers with a held-out check (the code-row variant) |
| `scripts/cli_proxy.py` | OpenAI-compatible endpoint over a Codex CLI subscription for every LLM role |

## Adapting to a task that is not τ²-bench

`examples/tau2/agent.py` wraps a `Runtime` as a tau2 agent: it builds `obs = {turn, policy, tools, conversation,
incoming}` from the environment and returns the harness `output` (a tool call or a message) to it. For another
environment, write the same wrapper: build `obs` the same shape, keep `memory` between turns, append every
turn's `routing` record and cost to a ledger, and write the step table with the episode reward attached
(`mixedharness/trace.py`). The compare script's table generation and the fit scripts read only those files.

## Credentials

`.env` in the checkout: `TYPESAFE_API_KEY` (or `TYPESAFE_API_KEY_PERSONAL`, used first), `ANTHROPIC_API_KEY`
for the API backend. With `--llm-backend codex` no LLM key is needed; start `scripts/cli_proxy.py` first.
