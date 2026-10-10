---
name: mixed-harness
description: Build and measure a Mixed Harness for a tool-using task: code lists the actions a state allows, Jev picks among them with a probability, the LLM takes the turns that are open or below threshold, and the thresholds are learned from the harness's own records. Use when a user has an agent that calls tools under a policy (customer service, operations, form-driven workflows) and wants it cheaper than an LLM-only agent without losing accuracy, or wants a routed Jev + LLM agent measured against an LLM-only baseline.
---

# Mixed Harness

Turn a tool-using task into a harness where every turn is taken by code, by Jev, or by an LLM, and the split
is read off records rather than written by hand. The package (`mixedharness/`) and the retail example
(`examples/tau2/`) are the reference; this skill tells you what to ask for, what to write, and what to run.

## Establish the task before building

Read the user's request and any project material first. Ask only for what is missing:

- **Tools and policy.** The tool schema (name, required arguments, types) and the policy document the agent must
  follow. These two are the only domain inputs the harness needs.
- **Observation and action.** What the harness sees each turn (the conversation so far, the event that just
  arrived: a user message or a tool result) and what it may return (one tool call with arguments, or one
  message). Which tools write to a database or have side effects: those are gated behind an explicit yes.
- **Scoring.** How an episode is judged (database end state, assertions, a human, a simulator with rewards),
  and whether a simulated user exists. Without a scorer there is no measurement and no refit.
- **An LLM-only baseline.** The agent the harness is measured against. If none exists, the benchmark's own
  agent or a plain tool-calling loop on the same model.
- **Credentials and budget.** `TYPESAFE_API_KEY` for Jev (direct API); an LLM API key, or a Codex CLI
  subscription through `scripts/cli_proxy.py` (`--llm-backend codex`). How many episodes may be run.

State the task contract in a few lines before writing code: inputs, actions, write tools, scorer, baseline,
budget.

## Build order

1. **Install.** `uv sync --locked --extra dev` in a checkout, or `uv pip install git+https://github.com/ShirleyHuang11/mixed_harness`
   into an existing environment. `uv run mixedharness doctor` checks the sandbox (`bwrap` on Linux,
   `sandbox-exec` on macOS) and which credentials are present. `uv run pytest` runs on mock providers.
2. **Write the option lister.** Copy the `compile` node of `examples/tau2/seeds.py` and replace the retail
   parts: how entities are parsed from the conversation and tool results, the legal-action catalog A(s) for
   each state (which tools can be called with arguments code can fill, which templated replies apply), the
   arguments code cannot fill (`needs`), the customer's request list (action intents minus completed writes),
   and the state type (`cluster`: a few fields such as authentication state, what just arrived, the status of
   the record in focus, the intent). Every write tool in the schema must appear in the catalog when its intent
   is present; a step whose catalog cannot cover the request stays open and goes to the LLM. Keep the Jev
   questions as they are: `route`, `action`, `action_alt`, `fits`, `user_confirms`, plus the task's candidate
   questions (which record, which variant, which payment method). See
   [references/integration.md](references/integration.md) for the exact contract of each node.
3. **Record the baseline.** Run the LLM-only arm on the task set. Its trajectories are the first learning
   material.
4. **Stage 0: label and replay.** `scripts/stage0_label_replay.py` labels each recorded turn closed or open by
   what the LLM did, scores the compiler's `closed` flag against the labels, and replays Jev on every turn to
   record its pick and probability. Check two numbers before going on: the share of closed turns (the ceiling
   of what Jev can take) and Jev's agreement with the LLM at probability 0.9 and above (should be near 90%).
   If the compiler's flag disagrees with the labels, fix the option lister first.
5. **Fit the thresholds.** `scripts/fit_policy.py --tau-only --replay <replay.jsonl> --out policy.json`: one
   threshold per state type (the lowest at which Jev's picks agreed with the LLM at least 90% of the time),
   `never` where none does.
6. **Run the three arms.** `scripts/tau2_compare.py` (or its equivalent for the task) runs LLM only, Jev only
   and the routed harness on the same tasks, same simulator, same seeds, and writes one table: tasks solved,
   cost per task relative to LLM only, who took the turns. Measure the noise first by running the same system
   twice; read accuracy differences only against that.
7. **Refit after every batch.** `scripts/fit_policy.py --tau-only --replay <replay.jsonl> --ledger-replay <routed ledgers.json ...>`
   adds the turns the LLM took (Jev's pick and probability recorded next to the LLM's action) and refits.
   `scripts/long_run.py --tau-only` runs the stream + held-out protocol: learn from a stream of conversations,
   score tasks the table never sees, one row per batch.
8. **Optional: code rows.** `examples/tau2/seeds_splits.py` and `scripts/learn_splits.py` learn rows where the
   records show the LLM always does the same thing for a state type plus one or two state features; those
   turns are then taken by code. Cheaper, more machinery; measure it the same way.

## What to report

The three-arm table with the noise band, the held-out curve if the long run was done, and, for every lost
episode class, which component produced the failing step (the attribution scripts in
`docs/design/tau2-four-arms.md` show how). Say what is in-sample. Never report a cost ratio without the
accuracy next to it.

## Things that went wrong before, so you do not repeat them

- A hand-written option catalog that omits a tool the policy allows: Jev then picks the nearest listed reply
  with high probability. Check catalog coverage against the schema.
- Offering a closing reply as soon as one write has completed when the customer asked for several things.
  Keep the request list.
- Asking Jev whether it can take a step and trusting the answer: it separates closed from open steps but not
  right from wrong picks. Gate on the probability of the pick.
- Reading Jev's pick from the final action in a record (100% agreement is the tell). Jev's pick is the decision
  node's action inside the recorded runtime result.
- Two runs started in the same second sharing a directory. Run directories are now created exclusively; keep
  it that way in any new runner.
