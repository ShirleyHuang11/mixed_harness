# Routed harness: reference

## The spec

A harness is a JSON graph (`mixedharness/spec.py`):

    {"format": "mixedharness/1", "name": "...", "jev_model": "jev-1.13.0", "llm_model": "claude-sonnet-5",
     "nodes": [...], "output": "<expr>", "memory": "<expr>"}

| Node | Fields | Reads |
|---|---|---|
| `python` | `source` (defines `run(obs, nodes, memory)`), `inputs` (node ids) | exactly its `inputs` |
| `expression` | `expression` | node ids referenced as `nodes['id']` |
| `jev` | `state` (expr), `questions` (map) or `questions_expression` (expr) | ids referenced in either |
| `llm` | `prompt` (expr; `null` skips the node), `schema` (JSON Schema subset), `max_tokens` | ids referenced in `prompt` |

Expressions are a restricted language over `obs`, `nodes` and `memory`: literals, indexing, arithmetic,
comparisons, conditionals, and a fixed set of functions (`len min max sum abs round mean median std last get
clip column text sorted any all`). Python nodes are validated against an allow-list before they run and
execute in a native sandbox (`bwrap` on Linux, `sandbox-exec` on macOS) with no network, no host environment
and memory, CPU and wall-time limits. `uv run mixedharness schema` prints the schema given to authoring models.

## One run

    from mixedharness.runtime import Runtime
    result = Runtime(spec, jev, llm).run(obs, memory)

Nodes run in dependency waves, independent nodes in parallel. `result['trace']` has one record per node
with its status (`ok`, `skipped`, `error`, `blocked`), timing, and everything the node saw and returned:
the Jev state, questions and response; the LLM prompt and reply; the python source hash and in-worker time.
A failing node raises `PipelineError` after the wave with the partial result attached; `infrastructure=True`
means a provider or sandbox outage rather than a candidate fault, and evolution treats the two differently.

Python nodes run in a warm worker: one sandboxed interpreter per pool slot, started on first use and reused.
Starting one costs a few hundred milliseconds on a network filesystem; a warm node costs a few milliseconds.

## Task description and triage

`mixedharness/triage.py` reads a TaskDescription (name, objective, observation fields, action space,
evaluator kind, decisions per episode, examples). `probe(task)` computes nine deterministic fields;
`triage(task, jev)` sends them, the objective and up to five clipped examples to Jev with the current
question map and returns the route (argmax of the `route` choice), the full distribution and the state sent.
`seed_for_route(route, task)` builds the seed:

- `settled`: python catalog → jev pick (`questions_expression`) → python decision
- `open_ended`: python prompt → llm draft (with the task's `output_schema`) → python validate
- `hybrid`: python prompt → llm propose k candidates → python criteria → jev select → python decision

## Evolution

    from mixedharness.evolve import run_evolution
    summary = run_evolution(train, validation, jev=..., llm=..., proposer=..., evaluator=..., store=...,
                            seed_spec=spec, task={...}, rounds=5, batch_size=2, budget=None)

The evaluator is task code: `evaluator(spec, episode, jev, llm=None, capture_traces=True)` returning
`{'episode_id', 'score', 'traces'}` with one trace per decision (the runtime result plus observation and
action). Each round GEPA selects a parent from the Pareto frontier over training instances, evaluates it on
a batch, hands the complete trajectories to the proposer (`mixedharness/reflect.py`), evaluates the proposed
spec on the same batch and keeps it only on strict improvement, then scores it on the full validation set.
Proposals that change `format`, `jev_model` or `llm_model`, or fail validation, are rejected and logged.
`budget={'latency_ms_p95': ..., 'cost_usd_per_episode': ...}` turns an over-budget episode into an invalid
candidate; usage is read from traces, never estimated.

Everything lands in the store (`mixedharness/store.py`): `run.json`, `events.jsonl`, `candidates/`,
`evaluations.jsonl`, `traces/`, `reflections/`, and GEPA's own log under `gepa/`.

## The routed run

    from mixedharness.routed import run_routed
    summary = run_routed(task, train, validation, jev=..., llm=..., proposer=..., store=..., evaluator=...,
                         task_id='classify', route=None, rounds=3, batch_size=2)

Probe, triage, seed, one evolution, then two records: `summary['route']` (what the triage chose) and
`summary['final_route']` (what the selected harness is made of). The pair is appended to
`<store>/triage/history.jsonl`, and `triage_store.improve` asks the proposer for a better question map,
replays the old and the new map through Jev over the whole history, and accepts the new one only if its
agreement with the recorded outcomes is strictly higher. Accepted maps are versioned in
`<store>/triage/questions.json`. Pass `route=` to override the triage; `improve_triage=False` skips loop 2.

## Freezing

`mixedharness/freeze.py` binds a spec to its hash, the sandbox interpreter identity, the evaluator's source
hash and the model ids. `evaluate_frozen` refuses to run when any of those changed.

## Errors, in one table

| Class | Meaning | Effect in evolution |
|---|---|---|
| `JevError`, `LLMError`, `SandboxUnavailable`, `ReflectionError` | provider, transport, sandbox or reflection-model failure | the run aborts; the store keeps everything so far |
| `NodeError`, `SourceError`, `ExpressionError`, `LLMOutputError`, `SchemaError` | the candidate's own fault | the episode scores `FAILURE_SCORE`; the run continues |
