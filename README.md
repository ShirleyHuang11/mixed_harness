# mixed_harness

Fork of [JevHarness](https://github.com/TianyuCodings/JevHarness) (`auto_jev`) extended with a routing layer:
a task is probed and triaged (settled / open-ended / hybrid), one harness is seeded from the chosen route's
template, and the existing GEPA evolution runs once. After every run the triage criteria get one
replay-validated improvement step.

- Design: `docs/superpowers/specs/2026-09-22-routed-harness-level1-design.md`
- Diagram: https://claude.ai/artifact/PDRJpw5s2nNLd4opTtxqmA

## Setup

    uv sync --locked --extra dev
    uv run pytest

Functional Python nodes run under `bwrap` on Linux and `sandbox-exec` on macOS.

## Routing

`auto_jev.routed.run_routed_evolution(task, train, validation, jev=..., llm=..., proposer=..., store=..., evaluator=..., task_id=...)`
probes the task, asks Jev for a route (settled / open_ended / hybrid), seeds one harness from that route's
template and runs the existing evolution once. Pass `route=` to override the triage. After each run the triage
criteria get one replay-validated improvement step. See `docs/routed-harness.md`.
