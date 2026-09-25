# τ²-bench: three arms, one environment

Design and decision table: `docs/design/tau2-three-arms.md`.

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
| `TYPESAFE_API_KEY` or `AI_GATEWAY_API_KEY` | Jev, for the `jev_only` and `harness` arms |
| `ANTHROPIC_API_KEY` | the `llm_only` arm, the harness llm nodes, and (by default) the user simulator and the NL-assertion judge |
| `OPENAI_API_KEY` | only if you pass `--user-llm gpt-4.1-2025-04-14`, tau2's official default |

## Run

    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --arms jev_only,harness --mock --num-tasks 1   # plumbing
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --num-tasks 5        # smoke
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --trials 3           # full

Artifacts land in `runs/tau2/<stamp>-<domain>-<split>/`: `manifest.json` (models requested and echoed,
seeds, task ids, credentials present by name, realised Jev/LLM stats), `<arm>/results.json` (tau2's own
format, so `tau2 view` and `tau2.metrics` work on it), `<arm>/ledgers.json` (one record per agent turn with
the complete runtime result), `summary.json` and `table.md` (generated from the artifacts).

## What each arm is

| Arm | Agent | Per turn |
|---|---|---|
| `llm_only` | tau2's stock `llm_agent` on `--agent-llm` (default `anthropic/claude-sonnet-5`) | 1 LLM call with policy, tools and full history |
| `jev_only` | `examples.tau2.seeds.jev_only_seed()` | 1 python node (facts + catalog + questions), 1 Jev request, 1 python node (decision) |
| `harness` | `examples.tau2.seeds.harness_seed()` | the same, plus two llm nodes that run only when the decision asks, and a merge node |

`examples/tau2/agent.py` wraps a `Runtime` as a tau2 `HalfDuplexAgent`; `examples/tau2/adapter.py`
exposes the same thing through the evaluator contract for evolution.

## Reading the table

- `pass^1` is tau2's reward (DB state plus assertions). `failed` counts simulations that never reached
  evaluation (infrastructure errors); they are not zeros.
- `turn latency` is wall time per agent turn. `model-only ms` sums the Jev and LLM node times; `sandbox ms`
  sums the sandboxed-Python node times. Python nodes run in a warm worker, so a turn costs a few
  milliseconds of sandbox time after the first one (mock plumbing on this cluster: 21 ms median per turn,
  p95 about 400 ms from the first-turn worker start). Both numbers are reported.
- The user simulator and the NL-assertion judge default to `anthropic/claude-sonnet-4-5-20250929` because
  the OpenAI account on this machine has no credit. The manifest records both the model used and tau2's
  default. Numbers from different user simulators are not comparable.
