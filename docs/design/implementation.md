# Implementation: an independent execution layer

Mixed Harness has its own implementation of the execution layer in `mixedharness/`. It follows the ideas of
[JevHarness](https://github.com/TianyuCodings/JevHarness) (a harness as a graph of sandboxed code and Jev
calls, evolved with GEPA over full traces) and is written from scratch: different spec format, different
runtime, different sandbox, different store. This file records what was worth keeping from JevHarness, what
was changed and why, and the order in which the package was built.

## What JevHarness gets right, and what we keep

| Kept as an idea | Where it lives here |
|---|---|
| The harness is data, not code: a JSON graph an authoring model can rewrite in full | `mixedharness/spec.py` |
| Code computes facts, Jev judges; several questions share one request | `runtime.py`, `jev.py` |
| Python nodes run in a native sandbox with no network, no host environment, hard limits | `sandbox.py` |
| Evolution reads complete traces, not summaries; proposals are whole specs; strict improvement on a shared batch; Pareto parent selection | `evolve.py` |
| A frozen harness binds the spec, the runtime and the evaluator by hash | `freeze.py` |
| The README shows a real recorded call and a latency table with sample counts, and says what a result does not establish | README, `docs/guides/tau2.md` |

## What is different

| Change | Reason |
|---|---|
| One spec format instead of four versions; python nodes declare `inputs`, the rest infer them | Less validator surface; nothing to keep backward-compatible |
| `llm` nodes are first-class; a `null` prompt skips the node | Code decides whether generation is needed; the skip is free |
| A warm sandbox worker per runtime instead of one process per node | Measured 0.5 s of process spawn per node on this cluster; a turn with three nodes spent more time spawning than judging |
| Jev over the direct TypeSafe API only, with a content-addressed cache | One transport to verify; the gateway was an extra failure mode |
| A flat JSON run store: events, candidates, traces, reflections | Enough for the evolution loop and for `table.md`; no resume protocol in the first version |
| No trading, dataset, paper-trading, server or website code | Not part of the idea |

## Build order

1. `expressions.py`: the restricted expression language (literals, `obs`/`nodes`/`memory`, arithmetic,
   comparisons, conditionals, a fixed set of functions). 2. `spec.py`: validation, hashing, the schema
   handed to the authoring model. 3. `sandbox.py`: source validation and the warm worker. 4. `runtime.py`:
   graph compilation, parallel execution, trace records. 5. `jev.py`, `llm.py`: providers with cache and
   mock. 6. `store.py`, `reflect.py`, `evolve.py`: the GEPA loop. 7. `triage.py`, `triage_store.py`,
   `routed.py`: routing (moved from the previous package). 8. `freeze.py`, `cli.py`. 9. Examples and the
   τ²-bench adapter switched over; the previous package removed.

Each step lands with its own tests on mock providers. The τ²-bench plumbing run under mock providers is the
acceptance test for the whole: same turn sequence as before, fewer milliseconds of sandbox time per turn.
