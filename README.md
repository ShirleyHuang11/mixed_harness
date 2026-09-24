# mixed_harness

A routed decision harness: code computes the facts, a judgment model (TypeSafe's Jev) makes the bounded
choices, and a language model writes only the text that has to be generated. The split is not hand-written.
A triage step reads the task description and picks one of three harness shapes, and the existing GEPA
evolution from [JevHarness](https://github.com/TianyuCodings/JevHarness) (`auto_jev`) then edits the whole
node graph, node kinds included. After every run the triage criteria take one replay-validated improvement
step.

- Design: `docs/superpowers/specs/2026-09-22-routed-harness-level1-design.md`
- Routed harness reference: `docs/routed-harness.md`
- τ²-bench three-arm test: `docs/tau2.md` and `docs/superpowers/specs/2026-09-24-tau2-three-arm-design.md`
- Interactive diagram: https://claude.ai/artifact/PDRJpw5s2nNLd4opTtxqmA

## How a task becomes a harness

```mermaid
flowchart LR
    T[TaskDescription<br/>objective, action space,<br/>evaluator, examples] --> P[probe<br/>code, 9 fields]
    P --> J{Jev triage<br/>settled / open_ended / hybrid}
    J -->|settled| S1[seed: python catalog<br/>→ jev pick → python decision]
    J -->|open_ended| S2[seed: python prompt<br/>→ llm draft → python validate]
    J -->|hybrid| S3[seed: llm propose k<br/>→ python criteria → jev select]
    S1 --> E[GEPA evolution, one run<br/>reflection edits the node graph<br/>and may swap node kinds]
    S2 --> E
    S3 --> E
    E --> B[selected harness<br/>runs/routed/id/decision.json]
    B --> R[record final_route vs triage_route<br/>runs/triage/history.jsonl]
    R --> I[improve triage questions<br/>replay old vs new on history,<br/>accept only if agreement rises]
    I -.-> J
```

The probe is deterministic and free: declared action kind, label count, whether examples carry a legal-action
catalog, evaluator exactness, expected-output shape. Jev sees the probe evidence, the objective and five
clipped examples, and returns a probability over the three routes. A wrong triage is corrected inside
evolution rather than by re-running it: the summary reports both the seed route and what the selected
harness ended up as.

## What one decision looks like

The same three node kinds run inside every harness. The example below is the τ²-bench retail seed
(`examples/tau2/seeds.py`), one agent turn:

```mermaid
flowchart LR
    O[observation<br/>policy, tool schemas,<br/>conversation, memory] --> F[facts · python<br/>parse ids and intents, track fetched<br/>entities, build the legal-action catalog<br/>and every question this turn might need]
    F --> Q[judge · jev, one request<br/>action choice + confirmation +<br/>which items / variant / payment]
    Q --> D[decision · python<br/>fill tool arguments, enforce<br/>confirm-before-write, pick a template]
    D -->|prompt = null| K[llm nodes skipped<br/>no call, no cost]
    D -->|template does not fit| L[draft · llm<br/>free-text reply or<br/>free-text arguments]
    K --> A[tool call or message]
    L --> A
    A --> M[(memory<br/>carried to next turn)]
    M -.-> O
```

Every Jev question is asked in one request (adding questions barely changes its latency), and an llm node
whose prompt evaluates to `null` is skipped by the runtime. The turn's ledger records what actually ran:
python node count, Jev calls, LLM calls and skips, tokens, and the model ids the providers echoed.

## The τ²-bench test: three arms, one environment

```mermaid
flowchart TB
    subgraph shared [same for all arms]
        ENV[tau2 retail environment<br/>114 tasks, 16 tools, DB-state reward]
        USR[same user simulator, same seeds]
    end
    ENV --> A1 & A2 & A3
    USR --> A1 & A2 & A3
    A1[llm_only<br/>tau2 stock agent<br/>1 LLM call per turn] --> T[table.md from artifacts:<br/>pass^1 · $/task · ms/turn ·<br/>LLM calls/turn · Jev calls/turn]
    A2[jev_only<br/>code + Jev + templates<br/>0 LLM calls] --> T
    A3[harness<br/>code + Jev + llm only when asked<br/>≤ 0.3 LLM calls per turn, predicted] --> T
```

The claim under test is not accuracy above the leaderboard. It is matched pass^1 against the same LLM used
alone, at a fraction of the cost and latency per turn, with the Jev-only arm showing where generation is
actually needed. Predictions, resolution and the decision table are written down before any run in the
design file. `docs/tau2.md` has the setup and the reading guide for the table.

## Setup

    uv sync --locked --extra dev
    uv run pytest

Functional Python nodes run under `bwrap` on Linux and `sandbox-exec` on macOS. Credentials go in a local
`.env` (see `.env.example`): `TYPESAFE_API_KEY` or `AI_GATEWAY_API_KEY` for Jev, `ANTHROPIC_API_KEY` for
llm nodes. `uv run auto-jev doctor` prints which are present.

## Routing

`auto_jev.routed.run_routed_evolution(task, train, validation, jev=..., llm=..., proposer=..., store=..., evaluator=..., task_id=...)`
probes the task, asks Jev for a route, seeds one harness from that route's template and runs the existing
evolution once. Pass `route=` to override the triage. See `docs/routed-harness.md`.

## Live run

    uv run python scripts/live_run.py --task classify --rounds 2 --train 6 --validation 6
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --num-tasks 5

## Repository map

| Path | What it holds |
|---|---|
| `auto_jev/` | spec validation, parallel runtime, Jev and LLM transports, triage, routed evolution, freezing |
| `examples/classify`, `examples/constrained_text` | bundled settled and open-ended tasks, run end-to-end under mock providers |
| `examples/tau2/` | tau2-bench adapter, the PipelineAgent wrapper, and the two seed harnesses |
| `scripts/` | `live_run.py` for a small real-model routed run; `tau2_compare.py` for the three-arm test |
| `docs/` | design files, the routed-harness reference, the tau2 guide |
