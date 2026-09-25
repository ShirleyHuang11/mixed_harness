# Mixed Harness: reference

## The spec

A harness is a JSON graph (`mixedharness/spec.py`):

    {"format": "mixedharness/1", "name": "...", "jev_model": "jev-1.13.0", "llm_model": "claude-sonnet-5",
     "nodes": [...], "output": "<expr>", "memory": "<expr>"}

| Node | Fields | Reads | Skips when |
|---|---|---|---|
| `python` | `source` (defines `run(obs, nodes, memory)`), `inputs` (node ids) | exactly its `inputs` | never |
| `expression` | `expression` | node ids referenced as `nodes['id']` | never |
| `jev` | `state` (expr), `questions` (map) or `questions_expression` (expr) | ids referenced in either | `questions_expression` is null |
| `llm` | `prompt` (expr), `schema` (JSON Schema subset), `max_tokens` | ids referenced in `prompt` | `prompt` is null |
| `agent` | `instructions`, `context` (expr → `{policy, tools, conversation}`), `max_tokens` | ids referenced in `context` | `context` is null |

A skipped node makes no call and outputs null. Expressions are a restricted language over `obs`, `nodes` and
`memory`: literals, indexing, arithmetic, comparisons, conditionals, and a fixed set of functions (`len min max sum
abs round mean median std last get clip column text sorted any all`). Python nodes are validated against an
allow-list before they run and execute in a native sandbox (`bwrap` on Linux, `sandbox-exec` on macOS) with no
network, no host environment and memory, CPU and wall-time limits. `uv run mixedharness schema` prints the schema
given to authoring models.

## One run

    from mixedharness.runtime import Runtime
    result = Runtime(spec, jev, llm).run(obs, memory)

Nodes run in dependency waves, independent nodes in parallel. `result['trace']` has one record per node
with its status (`ok`, `skipped`, `error`, `blocked`), timing, and everything the node saw and returned:
the Jev state, questions and response; the LLM prompt and reply; the python source hash and in-worker time.
A failing node raises `PipelineError` after the wave with the partial result attached; `infrastructure=True`
means a provider or sandbox outage rather than a candidate fault, and evolution treats the two differently.

Python nodes run in a warm worker: one sandboxed interpreter per pool slot, started on first use and reused.

## The routed harness

`examples/tau2/seeds.py` is the worked routed harness for a multi-step tool task. One turn:

    compile (python) → judge (jev) → route (python) → decision (python)
                                                     ↘ menu (llm) → menu_q (python) → menu_pick (jev) ↘
                                                     ↘ agent (agent)                                    → guard (python)

- **compile** is the action-space compiler: it parses the conversation, keeps fetched entities in memory, builds
  the legal-action catalog A(s) from state, fills what code can fill, lists the arguments it cannot fill
  (`needs`), flags the step `closed` (catalog non-empty and `needs` empty), names its `cluster`, and writes every
  question for the Jev request: `route` (closed / narrow / escalate), `action`, `action_alt` (the same options in
  reverse order, the agreement signal), `fits`, `user_confirms`, and the candidate questions.
- **route** applies the signals in order of authority: the compiler flag (an open step cannot be closed), Jev's
  route pick, the confidence threshold τ on the action pick (below it → escalate; 0.9 by default, from stage 0),
  agreement fall-through (closed with disagreement → narrow), the embedded per-cluster policy and its
  epsilon shadow share, trajectory-level escalation (k closed picks with disagreement in a row, or the same reply
  repeated). Without an LLM every path is forced closed and the pick is still recorded.
- **decision** is the closed path; **menu → menu_q → menu_pick** the narrow path (the LLM writes 2 to 4
  candidates, Jev picks); **agent** the escalate path. Paths not taken are skipped at no cost.
- **guard** assembles the action of the realised path, checks tool names and required arguments, gates database
  writes on every path behind an explicit yes to the same details, falls back to the closed decision when a menu
  or agent output is unusable, updates the trajectory counters, and returns the routing record the step table reads.

`mixedharness/router.py` holds the vocabulary (`PATHS`, `PATH_SOURCES`), the agreement helper, `route_metrics`
over step rows, and loop B: `fit_policy(steps)` gives each cluster the path with the best mean of
success − λ·cost once it has enough scored steps, and `policy_source_literal` renders it for embedding in a seed
(`routed_seed(policy=...)`), so the policy hashes with the harness version.

`mixedharness/trace.py` is the step table: `steps_from_ledger` turns an agent ledger into one row per turn
(cluster, flag, Jev's pick, realised path and why, agreement, confidence, action, guard, cost, latency, reward).
`mixedharness/mining.py` is loop A's macro-action miner: tool-call n-grams the LLM paths repeat, with support.

## Generic seeds

`mixedharness/seeds.py` reads a TaskDescription (name, objective, observation fields, action space, evaluator
kind, decisions per episode, examples) and builds a single-step seed: `seed_for_route('jev', task)` is catalog →
pick → decision; `seed_for_route('llm', task)` is prompt → draft, or context → agent for tool tasks.
`route_of_spec` reports what a finished spec is made of: jev, llm, jev+llm or code.

## Evolution

    from mixedharness.evolve import run_evolution
    summary = run_evolution(train, validation, jev=..., llm=..., proposer=..., evaluator=..., store=...,
                            seed_spec=spec, task={...}, rounds=5, batch_size=2, budget=None)

The evaluator is task code: `evaluator(spec, episode, jev, llm=None, capture_traces=True)` returning
`{'episode_id', 'score', 'traces'}` with one trace per decision (the runtime result plus observation and
action). Each round GEPA selects a parent from the Pareto frontier over training instances, evaluates it on a
batch, hands the complete trajectories to the proposer (`mixedharness/reflect.py`), evaluates the proposed
spec on the same batch and keeps it only on strict improvement, then scores it on the full validation set.
Proposals that change `format`, `jev_model` or `llm_model`, or fail validation, are rejected and logged.
`budget={'latency_ms_p95': ..., 'cost_usd_per_episode': ...}` turns an over-budget episode into an invalid
candidate; usage is read from traces, never estimated.

Everything lands in the store (`mixedharness/store.py`): `run.json`, `events.jsonl`, `candidates/`,
`evaluations.jsonl`, `traces/`, `reflections/`, and GEPA's own log under `gepa/`.

## Freezing

`mixedharness/freeze.py` binds a spec to its hash, the sandbox interpreter identity, the evaluator's source
hash and the model ids. `evaluate_frozen` refuses to run when any of those changed.

## Errors, in one table

| Class | Meaning | Effect in evolution |
|---|---|---|
| `JevError`, `LLMError`, `SandboxUnavailable`, `ReflectionError` | provider, transport, sandbox or reflection-model failure | the run aborts; the store keeps everything so far |
| `NodeError`, `SourceError`, `ExpressionError`, `LLMOutputError`, `SchemaError` | the candidate's own fault | the episode scores `FAILURE_SCORE`; the run continues |
