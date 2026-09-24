# Routed harness (Level 1)

## Flow

1. `triage.probe(task)` computes deterministic evidence from a `TaskDescription` (declared action kind,
   label count, whether examples carry a legal-action catalog, evaluator exactness, expected output shape).
2. `triage.triage(task, jev)` sends the evidence and up to five clipped examples to Jev with the current
   triage question map (initially `DEFAULT_TRIAGE_QUESTIONS`). The `route` is the argmax of the choice
   distribution; the distribution is recorded so a low-confidence call is visible, and the caller may pass
   `route=` to override.
3. `triage.seed_for_route(route, task)` returns a version-4 spec: `settled` → python catalog + jev pick +
   python decision; `open_ended` → python prompt + llm draft + python validate; `hybrid` → llm propose k
   candidates → python criteria → jev select → python decision.
4. `routed.run_routed_evolution(...)` runs `evolution.run_evolution` exactly once with that seed. A wrong
   triage is corrected inside evolution: the reflection model may swap node kinds. `final_route` in the
   summary reports what the selected candidate ended up as. Artifacts: `runs/routed/<id>/{decision,summary}.json`.
5. After the run, `triage_store.record` appends `{state, probabilities, triage_route, final_route → target}` to
   `runs/triage/history.jsonl` and `triage_store.improve` asks the reflection model for a better question map.
   Old and new maps are replayed through Jev on the whole history; the proposal is accepted only if its
   agreement with the targets is strictly higher. Accepted maps are versioned in `runs/triage/questions.json`.

## Version 4 specs

Adds `llm` nodes (`prompt_expression`, optional `output_schema`, optional `max_tokens`) and top-level
`llm_model`. Reflection may swap node kinds and insert generate-then-select structures; it may not change
`jev_model`, `llm_model` or the evaluator.

## LLM nodes that decide not to run

An `llm` node whose `prompt_expression` evaluates to `null` is skipped: no call, no cost, output `null`, trace
status `skipped`. Code decides whether generation is needed; downstream nodes must treat `nodes['<id>']`
as possibly `null`. A prompt that evaluates to anything else must be a nonempty string.

## LLM errors

`LLMProviderError` (missing key, timeout, HTTP failure, cache-only miss, cache corruption) is tagged
`task_infrastructure_error` and stops an evolution run like a Jev outage. `LLMOutputError` (empty, unparsable or
schema-violating answer) is a candidate-level node error and only scores that candidate down.

## Budgets

`run_evolution(..., budget={'latency_ms_p95': ..., 'cost_usd_per_episode': ...})`. Usage is derived from
complete traces (`budget.episode_usage`). A scored episode over budget becomes an invalid candidate
(`status='over_budget'`, score `-1e9`). Unpriced calls are counted, not guessed.

## Linux

Functional Python nodes run under `bwrap`: empty root, read-only `/usr` and the interpreter prefix,
no network, no host environment, `RLIMIT_AS`/`RLIMIT_CPU`/wall-time limits. Missing `bwrap` or refused
user namespaces raise `PythonSandboxUnavailable` (an infrastructure error, never a candidate score).
bwrap 0.4 has no `--clearenv`; the worker is launched with an explicit minimal environment instead.

## Examples

    uv run python -c "from examples.classify import adapter as a; print(a.load('train')[0]['name'])"

`examples/classify` (settled, labels) and `examples/constrained_text` (open-ended, code-checked) run
end-to-end with mock providers in `tests/test_routed.py`.

## Live run

With `TYPESAFE_API_KEY` (direct, from https://console.typesafe.ai/keys) or `AI_GATEWAY_API_KEY` (Vercel AI Gateway),
plus `ANTHROPIC_API_KEY` for llm nodes, in `.env`:

    uv run python scripts/live_run.py --task classify --rounds 2 --train 6 --validation 6
    uv run python scripts/live_run.py --task classify --jev-transport typesafe --jev-model jev-latest

Reflection uses the local `claude` CLI (`configs/claude_local.json`). Responses are cached under
`.cache/{jev,llm}/<namespace>/`; reuse `--namespace` to replay. Outputs land in `runs/routed/<id>/`
(`decision.json`, `summary.json`), `runs/<run_id>/` (events, reflections, candidates) and `runs/triage/`.
