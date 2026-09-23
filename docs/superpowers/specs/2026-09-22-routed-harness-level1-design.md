# Routed Harness — Level 1 Design

Companion diagram: https://claude.ai/artifact/PDRJpw5s2nNLd4opTtxqmA

## Goal

Extend JevHarness (`auto_jev`) so that, for a new task, the system first decides whether the
task has a settled (enumerable) action space or an open-ended output, seeds **one** harness from
the chosen route's template, and evolves it with the existing GEPA loop unchanged. Re-routing
happens inside that single evolution: the reflection model may swap node kinds. Level 2 (evolving
the improver itself) is out of scope; Level 1 must ship as working, tested software on its own.

## Non-goals

- No general Level 2 / meta-evolution. The one improver component that does learn is the triage question map (see §8); reflection instructions and seed templates stay fixed.
- `llm` nodes do not call tools; text in, schema-validated JSON (or plain text) out.
- No Pokémon / Node.js dependency in this repo. Pokémon stays upstream.
- No changes to GEPA internals; exactly one `run_evolution` per task. No multi-seed, no cross-run selection.

## Platform

- Python 3.11+, managed with `uv`. Runs on Linux (this HPC) and macOS.
- Functional Python nodes must run on Linux through `bwrap` (bubblewrap) with the same
  fail-closed semantics as the existing macOS `sandbox-exec` path. No host `exec` fallback.
- No API keys are assumed present; every feature is testable with mock providers.

## Vocabulary

- **route** ∈ {`jev`, `llm`, `hybrid`, `code`} — derived from a spec's node kinds:
  jev nodes only → `jev`; llm nodes only → `llm`; both → `hybrid`; neither → `code`.
- **RouteDecision** — output of triage: `{route, probabilities, confidence, evidence, questions,
  response}` where `route` ∈ {`settled`, `open_ended`, `hybrid`} (triage vocabulary) is the argmax.
  Mapping triage→seed: `settled`→jev seed, `open_ended`→llm seed, `hybrid`→hybrid seed.
  Low confidence is recorded, not acted on: the caller may override `route`.
- **TaskDescription** — JSON given by the user/agent:
  ```
  {"name": str, "objective": str,
   "observation_fields": [str],                 # keys present in obs
   "action_space": {"kind": "labels"|"catalog"|"open"|"declared_unknown",
                    "labels": [str]?,            # for kind=labels
                    "catalog_field": str?},      # obs key holding legal action list, kind=catalog
   "evaluator": {"kind": "exact_match"|"environment"|"code_check"|"reward_model"|"judge"},
   "decisions_per_episode": int, "latency_budget_ms": number|null,
   "output_schema": {...}?,                      # for open outputs
   "examples": [{"observation": {...}, "expected": any}]}
  ```

## Components

### 1. Linux sandbox (`auto_jev/python_nodes.py`)
`_sandbox_spec()` returns `{command, executable, sandbox_executable, profile, interpreter_flags}`.
On darwin it builds the existing `sandbox-exec` command. On linux it builds a `bwrap` command:
`--unshare-all --die-with-parent --new-session --clearenv`, read-only binds of `/usr`,
`/etc/ld.so.cache` (if present) and `sys.base_prefix`, symlinks `/lib`,`/lib64`,`/bin` → `usr/*`,
`--proc /proc --dev /dev --tmpfs /tmp --chdir /tmp`, then `<python> -I -S -B`.
Bootstrap sets `RLIMIT_AS` in addition to `RLIMIT_DATA` on Linux (kernel-enforced memory cap);
RSS polling is darwin-only. Any other platform, or missing `bwrap`, raises
`PythonSandboxUnavailable`. `runtime_identity()` on Linux hashes the interpreter, `bwrap`,
the argument profile, stdlib `.py` tree, `lib-dynload`, and `/etc/os-release`.
Known limitation (documented): bwrap cannot deny reading pre-existing `.pyc`; identity binds `.py`.

### 2. LLM provider (`auto_jev/llm.py`, `auto_jev/schema.py`)
`LLMClient(kind='mock'|'anthropic'|'openai', model, api_key_env=None, cache_dir=None,
cache_namespace=None, cache_only=False, timeout=60, max_tokens=1024, mock_responder=None)`.
`complete(prompt: str, output_schema: dict|None=None, model=None) -> dict` with keys
`text, value, model, usage, elapsed_ms, cache_hit, request_hash, cost_usd (None if unknown),
client_metadata`. When `output_schema` is given, `text` must parse as JSON and validate
(`schema.validate_json_schema`), else `LLMProviderError`. Cache format mirrors `JevClient`
(content-hashed file with `response_digest`). Mock default: builds the minimal instance of the
schema (required fields with type defaults) or echoes the prompt's last line as text;
`mock_responder(prompt, schema) -> str` overrides. `stats` = calls, cache_hits, input_tokens,
output_tokens, elapsed_ms, cost_usd, unpriced_calls.
`validate_json_schema(value, schema)` supports: type (object/array/string/number/integer/boolean/
null and lists), properties, required, additionalProperties(false), items, enum, minimum,
maximum, minLength, maxLength, minItems, maxItems. Anything else raises `ValueError('unsupported')`.

### 3. Spec v4 (`spec.py`, `flow.py`, `runtime.py`)
Version 4 = version 3 + `llm` nodes + optional top-level `llm_model` (required when any llm
node exists). llm node fields: `id, kind:"llm", prompt_expression (restricted expression →
string), output_schema (dict, optional), max_tokens (int, optional), depends_on (optional)`.
Dependencies: inferred from `prompt_expression` plus `depends_on`. Runtime:
`PipelineRuntime(spec, jev, llm=None, max_workers=8)`; constructing with llm nodes and
`llm=None` raises `ValueError`. Node record adds `prompt`, `output_schema`, `response`;
result = `response['value']` if schema else `response['text']`. `pipeline_schema(4)` describes it.
All `version==3` checks become `version>=3`.

### 4. Budgets (`evolution.py`, new `auto_jev/budget.py`)
`episode_usage(result) -> {latency_ms_p95, latency_ms_median, decisions, cost_usd|None,
unpriced_calls}` computed from `result['traces'][*]['elapsed_ms']` and each jev/llm node
response's cost. `check_budget(usage, budget) -> list[str]` (violations); budget keys
`latency_ms_p95`, `cost_usd_per_episode`. `run_evolution(..., budget=None, llm=None)`:
an evaluated episode that violates budget gets `score=FAILURE_SCORE`, `status='over_budget'`,
`error` naming the violation; usage is stored in the evaluation record and in `config['budget']`.
Evaluators that declare an `llm` parameter receive it. `evaluate_frozen(..., llm=None)` same.

### 5. Triage (`auto_jev/triage.py`)
- `probe(task: TaskDescription) -> evidence` (pure code): `declared_action_kind`,
  `label_count`, `catalog_present_in_examples` (every example obs has a non-empty list at
  `catalog_field`), `exact_validator` (evaluator kind ∈ {exact_match, environment, code_check}),
  `expected_kind` (`label` if every expected ∈ labels; `short_text` if str with median ≤ 12
  whitespace tokens; `long_text` if str otherwise; `structured` if dict/list), `median_expected_tokens`,
  `decisions_per_episode`, `latency_budget_ms`, `example_count`.
- `DEFAULT_TRIAGE_QUESTIONS` — a Jev question map: `route` (choice: settled / open_ended / hybrid),
  `code_computes_facts` (noul), `needs_multistep` (noul). This is only the *initial* version; the
  live version is stored and improved per §8.
- `triage(task, jev, *, questions=None) -> RouteDecision`. `route` = argmax of the `route`
  choice probabilities, `confidence` = that probability. A `choice` outside the three routes
  raises `ValueError`. State sent to Jev = evidence + up to 5 examples truncated to 300 chars
  each + objective.
- `seed_for_route(route, task) -> PipelineSpec(v4)`; three templates:
  - `settled`: python `catalog` node builds `{"action": choice question over labels or
    obs[catalog_field]}`; jev `pick` node with `questions_expression`; python `decision` returns
    the chosen id (validated against the option list, else first option).
  - `open_ended`: python `prompt` node builds a prompt string from `observation_fields`; llm
    `draft` node with `task['output_schema']` (or none); python `validate` node returns the value.
  - `hybrid`: python `prompt` node asks for `k=3` candidates as `{"candidates": [str]}`; llm
    `propose`; python `criteria` node turns candidates into a choice question; jev `select`;
    python `decision` returns the selected candidate text.
  Every template passes `validate_spec` and runs end-to-end under mock providers.
- `route_of_spec(spec) -> 'jev'|'llm'|'hybrid'|'code'`.

### 6. Routed evolution (`auto_jev/routed.py`)
`run_routed_evolution(task, train, validation, *, jev, llm, proposer, store, evaluator, task_id,
route=None, seed_pipeline=None, budget=None, evolution_rounds=None, max_metric_calls=None,
reflection_batch_size=None, seed=0, task_context=None, task_contract=None, triage_questions=None)
-> summary`:
1. `decision = triage(task, jev)`; `route = route or decision['route']` (explicit override wins and
   is recorded as `route_source: 'override'`, otherwise `'triage'`).
2. `spec = seed_pipeline or seed_for_route(route, task)`; if it has llm nodes, `spec['llm_model']`
   is set to `llm.model`.
3. One `run_evolution(..., seed_pipeline=spec, run_name=f'{task_id}-{route}', llm=llm,
   budget=budget, task_context={**task_context, 'route': route, 'triage': decision-without-response})`.
4. Return `{**run_summary, 'decision': decision, 'route': route, 'route_source', 'seed_hash',
   'seed_route': SEED_ROUTE_OF[route], 'final_route': route_of_spec(best candidate)}` and write it
   to `<store>/routed/<task_id>-<ts>/summary.json` (plus `decision.json`).
5. Append a triage history record and attempt one triage improvement (§8). The summary gains
   `triage_improvement` (the §8 result or `{'status': 'skipped', 'reason': ...}`).
The triage step in 1 uses `triage_store.current_questions(store)` unless `triage_questions` is given.
Reflection instruction (in `evolution.py`) gains one sentence for v4: node kinds may be swapped
(jev↔llm) and a generate-then-select structure may be inserted, within the stated budget — this is
how a wrong triage gets corrected without a second run.

### 8. Triage improvement (`auto_jev/triage_store.py`)
The triage question map is the one improver component that learns, one bounded step per routed run,
validated by replay against accumulated history. Files under `<store>/triage/`:
- `questions.json` — `{version, questions, accepted_at, agreement, history_size}`; absent → `DEFAULT_TRIAGE_QUESTIONS`, version 0.
- `history.jsonl` — one record per routed run: `{routed_id, task_id, task_name, state (exactly what
  triage sent to Jev), probabilities, triage_route, chosen_route, route_source, final_route,
  target (final_route mapped to triage vocabulary), best_score, questions_version, recorded_at}`.
  Mapping `jev→settled`, `llm→open_ended`, `hybrid→hybrid`, `code→settled`.
- `improvements.jsonl` — one record per attempt: proposal hash, agreement before/after, accepted, reason.

API:
- `current_questions(store) -> (questions, version)`.
- `record(store, summary) -> history record`.
- `replay_agreement(questions, history, jev) -> float` — for each history record call
  `jev.judge(record['state'], questions)` and take the `route` argmax; agreement = mean of
  `argmax == target`. Empty history → 0.0.
- `improve(store, jev, proposer, *, min_history=2) -> result`. Skips (`status: 'skipped'`) when
  history has fewer than `min_history` records. Otherwise builds a JSON prompt
  `{instruction, current_questions, history, vocabulary: ROUTES}` asking for a complete replacement
  question map (same three question ids, `route` criteria keys exactly `settled/open_ended/hybrid`,
  strict `validate_questions`). Invalid proposal → `status: 'rejected', reason`. Valid → compute
  `before = replay_agreement(current, history, jev)` and `after = replay_agreement(proposed, history, jev)`;
  accept iff `after > before`; on accept write `questions.json` with `version + 1`. Prompt and
  response are archived under `<store>/triage/reflections/NNNN-<sha>.{prompt.json,response.txt}`.
  Proposer exceptions are recorded as `status: 'error'` and never abort the routed run.
### 7. Example adapters (`examples/classify/`, `examples/constrained_text/`)
- `classify.adapter.evaluate_episode(spec, episode, jev, llm=None, *, capture_traces=True)`:
  obs = `{"text": episode["text"], "labels": episode["labels"]}`; output must be one label;
  score = 1.0 if equal to `episode["label"]` else 0.0. Ships a 40-row fixture
  (`examples/classify/fixture.json`, banking-style intents) and a `task.json` TaskDescription.
- `constrained_text.adapter.evaluate_episode(...)`: obs = `{"instruction", "constraints"}`;
  output is `{"text": str}`; score = fraction of constraints satisfied, constraints kinds:
  `max_words`, `min_words`, `must_include`, `must_not_include`, `ends_with`. Fixture of 20 rows
  + `task.json`.
Both are importable, have inspectable source (for `build_task_contract`), and raise
`task_infrastructure_error`-tagged exceptions only for infrastructure.

## Acceptance

- `uv run pytest` green on Linux, including `tests/test_python_nodes.py` under bwrap.
- `examples/classify` triages to `settled` with a scripted Jev and evolves 1 round under mock
  providers via `run_routed_evolution`, producing `summary.json` with `seed_route == 'jev'`.
- `examples/constrained_text` triaged to `open_ended` runs the llm seed; the same task with
  `route='hybrid'` override runs the hybrid seed and records `route_source == 'override'`.
- A frozen artifact from a routed run passes `validate_artifact` and `evaluate_frozen` with
  the same mock providers.
- After two routed runs, `triage_store.improve` with a scripted proposer whose criteria make the
  replay agree better is accepted (version 1); a proposal that agrees worse is rejected and the
  stored questions are unchanged.
