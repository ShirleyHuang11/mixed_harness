# τ²-bench: three arms, one environment

Design, stages and decision table: `docs/design/tau2-four-arms.md`.

## Setup

tau2-bench needs Python 3.12; this repo runs on 3.11+. The two share one virtualenv: tau2's, with this repo
installed into it as an editable package.

    git clone https://github.com/sierra-research/tau2-bench ~/tau2-bench
    cd ~/tau2-bench && uv sync && uv pip install websockets && uv pip install -e ~/mixed_harness

`websockets` is imported at load time by tau2's voice path even without the voice extra; the voice extra
itself needs `portaudio`, which this cluster does not have. Set `TAU2_HOME` if the checkout lives elsewhere.

Credentials, read from `mixed_harness/.env`:

| Variable | Used by |
|---|---|
| `TYPESAFE_API_KEY` | Jev, for the `jev_only` and `routed` arms and for stage 0; `TYPESAFE_API_KEY_PERSONAL` is used first when set |
| `ANTHROPIC_API_KEY` | with the default API backend: the `llm_only` arm, the harness agent node, the user simulator and the NL-assertion judge |
| none | with `--llm-backend codex`: every LLM role runs through a Codex CLI subscription; start `scripts/cli_proxy.py` first and pass `--proxy-base` |
| `OPENAI_API_KEY` | only if you pass `--user-llm gpt-4.1-2025-04-14`, tau2's official default |

## Run

    ~/tau2-bench/.venv/bin/python scripts/stage0_label_replay.py                                    # stage 0: labels + Jev replay on recorded LLM-only traces
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --arms jev_only,routed --mock --num-tasks 1   # plumbing (needs a user simulator with credit)
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --num-tasks 5   # smoke
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base                 # stage 2
    ~/tau2-bench/.venv/bin/python scripts/cli_proxy.py --backend codex --model gpt-5.5 --port 8765 &    # subscription instead of an API
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --arms routed --policy <policy.json> --llm-backend codex --proxy-base http://127.0.0.1:8765/v1
    ~/tau2-bench/.venv/bin/python scripts/fit_policy.py --replay runs/stage0/<stamp>/replay.jsonl --steps <routed steps.jsonl ...> --ledgers <routed ledgers.json ...> --out policy.json
    ~/tau2-bench/.venv/bin/python scripts/long_run.py --passes 6 --replay runs/stage0/<stamp>/replay.jsonl --llm-only-run runs/tau2/<llm_only run>   # stream + held-out curve

Artifacts land in `runs/tau2/<stamp>-<domain>-<split>/`: `manifest.json` (models requested and echoed, seeds,
task ids, credentials present by name, realised Jev/LLM stats), `<arm>/results.json` (tau2's own format, so
`tau2 view` and `tau2.metrics` work on it), `<arm>/ledgers.json` (one record per agent turn with the complete
runtime result), `<arm>/steps.jsonl` (the step table: one row per turn with the routing record and the episode
reward), `summary.json` and `table.md` (generated from the artifacts). Stage 0 writes `runs/stage0/<stamp>/`
with `labels.jsonl` (hand-check list), `replay.jsonl`, `report.json`, `report.md`.

## What each arm is

| Arm | Agent | Per turn |
|---|---|---|
| `llm_only` | tau2's stock `llm_agent` on `--agent-llm` (default `anthropic/claude-sonnet-5`, or `openai/gpt-5.5` through the proxy with `--llm-backend codex`) | 1 LLM call with policy, tools and full history |
| `jev_only` | `examples.tau2.seeds.jev_only_seed()` | compile (python), 1 Jev request, route, decision, guard (python); no LLM |
| `routed` | `examples.tau2.seeds.routed_seed(policy=...)` with the table from `--policy` | the same; a code row takes the step with no model call, Jev's pick is kept when its probability clears the cluster's τ, otherwise agent (one native tool-calling turn) |
| `routed_evolved` | `--evolved-spec` (a candidate from an evolution run) or the routed seed with a table fitted on `--policy-steps` | as routed |

`examples/tau2/agent.py` wraps a `Runtime` as a tau2 `HalfDuplexAgent`; `examples/tau2/adapter.py`
exposes the same thing through the evaluator contract for evolution.

## Reading the table

- `pass^1` is tau2's reward (DB state plus assertions). `failed` counts simulations that never reached
  evaluation (infrastructure errors); they are not zeros.
- `turn latency` is wall time per agent turn. `model-only ms` sums the Jev and LLM node times; `sandbox ms`
  sums the sandboxed-Python node times. Python nodes run in a warm worker, so a turn costs a few
  milliseconds of sandbox time after the first one.
- The routing table (from `steps.jsonl`): `Jev share` is the fraction of steps whose realised path was closed
  (a code row or an accepted Jev pick); `escalate` the fraction the LLM took; `low conf` the share where Jev's
  probability on its pick was below the cluster's τ (or the cluster is marked never); `code override` the share
  where the compiler flagged the step open (no listed option covers the request, or a request remains with only
  replies listed); `guard` the count of guard interventions (write gate, bad arguments, unusable agent output).
  `summary.json` carries the same by source (`structural`, `learned`, `jev`, `low_confidence`, `code_override`,
  `trajectory`), which is where the code / Jev / LLM shares in the README come from.
- The user simulator and the NL-assertion judge default to `anthropic/claude-sonnet-4-5-20250929` (tau2's
  official gpt-4.1 was not available here); with `--llm-backend codex` they run on gpt-5.5 through the proxy.
  The manifest records the model used and tau2's default. Numbers from different user simulators are not
  comparable; the README's table is the gpt-5.5 environment. Two runs launched in the same second get
  distinct directories (`<stamp>-1-...`).
- A cost under `--llm-backend codex` is imputed: the subscription bills nothing, so tokens are priced at
  gpt-5.5's list price for every arm alike (`cost_imputed` in the manifest).
