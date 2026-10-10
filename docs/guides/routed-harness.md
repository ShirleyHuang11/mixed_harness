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

`examples/tau2/seeds.py` is the worked routed harness for a multi-step tool task (the one in the README
figures). One turn:

    compile (python) → judge (jev) → route (python) → decision (python)                                  ↘
                                                     ↘ menu (llm) → menu_q (python) → menu_pick (jev) ↘    → guard (python)
                                                     ↘ agent (agent)                                    ↗

- **compile** is the action-space compiler: it parses the conversation, keeps fetched entities in memory,
  keeps the customer's request list (action intents minus completed writes), builds the legal-action catalog
  A(s) from the state, fills what code can fill, lists the arguments it cannot fill (`needs`), flags the step
  `closed` (catalog non-empty, `needs` empty, the request covered, and after a write no request left with
  only replies listed), names its `cluster` (state type), and writes every question for the Jev request:
  `route` (closed / escalate; `narrow` only when the menu path is on), `action`, `action_alt` (the same
  options in reverse order, the agreement signal), `fits`, `user_confirms`, and the candidate questions.
- **route** applies the signals in order of authority: the compiler flag (an open step cannot be closed),
  Jev's route pick, the threshold τ for the cluster on the action pick (`policy["tau"]`; 0.9 by default,
  NEVER for clusters the table marks for the LLM), the embedded policy's epsilon shadow share, and
  trajectory-level escalation (the same reply repeated). Without an LLM every path is forced closed and the
  pick is still recorded.
- **decision** is the closed path; **menu → menu_q → menu_pick** the narrow path (off by default: the LLM
  writes 2 to 4 candidates, Jev picks); **agent** the escalate path. Paths not taken are skipped at no cost.
- **guard** assembles the action of the realised path, checks tool names and required arguments, gates
  database writes on every path behind an explicit yes to the same details, falls back to the closed decision
  when a menu or agent output is unusable, updates the trajectory counters, and returns the routing record the
  step table reads.

`mixedharness/router.py` holds the vocabulary (`PATHS`, `PATH_SOURCES`), `route_metrics` over step rows, and
the table's learning rules: `fit_tau` (a threshold per cluster from agreement rows: the stage-0 replay plus,
through `scripts/fit_policy.py --ledger-replay`, every step the LLM took in recorded ledgers, where Jev's pick
and probability sit next to the LLM's action), `merge_policy`, `compact_policy`, and `policy_source_literal`,
which embeds the table in the seed so it hashes with the harness version (`routed_seed(policy=...)`).
`scripts/fit_policy.py --tau-only` writes the table.

`examples/tau2/seeds_splits.py` is the variant with code rows: its compiler also computes a few state features
and consults learned rows (`policy["defaults"]`, `policy["splits"]`) before asking Jev; `fit_from_steps` and
`scripts/learn_splits.py` learn those rows. The README's last results section measures it.

`mixedharness/trace.py` is the step table: `steps_from_ledger` turns an agent ledger into one row per turn
(cluster, flag, who took the step and why, probability, action, guard, cost, latency, reward).
`mixedharness/mining.py` is the macro-action miner: tool-call n-grams the LLM paths repeat, with support.

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
