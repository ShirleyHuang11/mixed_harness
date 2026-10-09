# Mixed Harness

An agent that serves customers, calls tools and follows a policy makes a decision on every turn. Most of
those decisions are small: which record to look up, which of four items the customer meant, whether they
just said yes. A standard LLM agent pays a full model call for each one, re-reading the policy, the tool
list and the whole conversation every time: about 8,300 input tokens and 2 seconds per turn on τ²-bench
retail. Some decisions are not small: an explanation the policy does not script, a plan across several tool
calls, a reply that has to be composed. For those, the full model call is the right tool.

Mixed Harness splits a tool-using task along that line with one table, and learns the table from its own
records. **For every type of state the table says who takes the step: code, Jev (TypeSafe's judgment model,
400 ms, typed output, a probability with every answer), or an LLM. Jev takes a step only when its
probability on its own pick clears the threshold the table holds for that state. Every step is recorded, and
after every batch of conversations the table is recomputed: a state type moves to code when the LLM always
did the same thing there, to Jev when Jev's picks match the LLM's at some probability, and stays with the
LLM otherwise.** On τ²-bench retail the harness runs at 0.62× the cost of the LLM-only agent, 1.8 points
below it on pass^1 (noise is 9 points), and on conversations it has never seen its cost falls from 0.75× to
0.60× of the LLM's over the first 304 conversations it serves.

## Architecture

![Runtime path: task state, compiler, Jev router, the paths, guard, trace store, and the loops feeding a new version back](docs/media/fig_runtime_path.svg)

**Runtime, every step, no domain code in the picture.** Code lists the options the state allows: the tools
whose arguments can be filled, the known entities (orders, items, payment methods), the templates, and
"none of these". The table is looked up by state type. Code takes the step with no model call. Jev takes it
with one request, which also asks the candidate questions (which item, which variant, did the customer
confirm), and the pick executes when Jev's probability on it is at or above the threshold. Otherwise the LLM
takes the step in one native tool-calling turn. The guard checks tool names and arguments, lets no database write through
without the customer's yes to that operation, and falls back to the listed default when a model's output is
unusable. The step is written to the step table with the state type, who took it, the probability, the action
and later the episode's reward.

**Offline, after every batch of conversations, the only place domain information enters.** Two products.
The option-listing code is written once by a language model from the tool schema, the policy and the
patterns in the records, then frozen and run as ordinary code; in the current version it is hand-written
for retail. The table is recomputed from the records by four rules. A state type goes to code when the
LLM's action there was the same at least 90% of the time. A state type too coarse for that is split by one
or two cheap state features (is the profile loaded, how many orders are loaded, which tool just returned,
did the customer just say yes) until a subset of at least ten steps has one action at 90%, and that subset
becomes a code row. A state type goes to Jev with threshold τ when Jev's picks at or above τ agree with the
LLM's at least 90% of the time. A state type goes to the LLM when no τ reaches that. Traffic only moves
toward the cheaper end, and the guard and the thresholds hold accuracy while it moves.

![Evolution loops: the trace store feeds the execution layer (slow) and the routing layer (fast); both deploy a new harness version](docs/media/fig_evolution_loops.svg)

**Why Jev shapes the design.** Jev returns typed outputs only, so code or an LLM fills free text. Its
probability is calibrated: on 1,002 recorded closed steps it agreed with the LLM on 98% of picks at 0.99 or
above, 82% between 0.90 and 0.99, and 35% below 0.90, so the probability is the abstain signal. Asked instead
which path a step should take, it separated closed from open steps well enough (85% of its "closed" calls were
closed) but its pick on those steps matched the LLM's action only 52% of the time, so the route question was dropped. It picks well when the
options are a judgment and poorly when the next step depends on a plan the conversation implies (44% in the
clusters right after a lookup), so those state types sit with the LLM until the records show otherwise.

## The table on retail

The table has three kinds of rows, all computed from records: 1,389 LLM-only turns for the thresholds, and
the routed conversations served so far for the code rows. After 304 conversations the learned part has 30
rows. A sample, with the number of recorded steps each row was learned from and the share of them on which
the LLM took that action:

| state type | plus | → | n | share |
|---|---|---|---|---|
| authenticated, a tool just returned, order cancelled, cancel intent | the tool was cancel_pending_order | report the result | 22 | 95% |
| authenticated, a tool just returned, no order in focus, exchange intent | profile not loaded yet | load the profile | 19 | 100% |
| authenticated, a tool just returned, no order in focus, exchange intent | the tool was get_user_details | load the next order | 19 | 95% |
| authenticated, customer spoke, order pending, cancel intent | not a question, not a yes | report the state | 18 | 100% |
| not authenticated, one name and a zip code | | look the customer up | 79 | 97% |
| authenticated, a lookup just returned, no order in focus (otherwise) | | LLM, never Jev | | Jev agrees 52% to 71% |
| everything else | | Jev at τ = 0.9 | | 92% agreement at ≥ 0.9 |

Of the 30 learned rows, 16 report a tool result or the current state, 8 load an order, 3 load the profile
and 3 perform a return, exchange or modification. Rules that read as natural but the records contradicted
were tried and removed: "report a result as soon as it arrives" (the LLM did that 67% of the time under one
simulator), "load the profile after authentication" (23%), "ask which order when several are known" (11%).
A row enters the table only when the records agree with it, and leaves when they stop agreeing.

**What Jev is for.** Jev decides the steps where the options are listed and the arguments are known but a
judgment remains: which record to load next, which variant the customer means, which payment method, whether
the reply is a yes. It takes 23% of all steps in every version measured, and that share does not change as
the table learns: the learned rows take steps from the LLM, not from Jev. Jev is not asked whether it can
take a step. It is asked to pick, and the pick is used when its probability clears the threshold.

## Measuring it

![The tau2-bench test: four arms on the same benchmark, one table](docs/media/fig_measurement.svg)

The benchmark is τ²-bench retail: 114 customer-service conversations (returns, exchanges, cancellations,
address changes) with 16 tools and a simulated customer. A task counts as solved when the database ends in
the right state and the required things were said. Every system below runs the same 114 tasks against the
same simulated customer (gpt-5.5) with the same seed. The decision rules, predictions and every result are in
`docs/design/tau2-four-arms.md`.

How to read the table:

- **tasks solved** is the fraction of the 114 tasks solved (pass^1). Running the same system twice moves
  this number by up to 9 points, so differences smaller than that are noise.
- **cost** is what the agent spent on model calls, relative to the LLM-only agent (1.00×). Jev is priced at
  about one hundredth of an LLM call, so this column is almost entirely LLM calls.
- **who took the steps** is the share of all conversation turns decided by code (no model call), by Jev, and
  by the LLM.

| system | tasks solved | cost | who took the steps: code / Jev / LLM |
|---|---|---|---|
| LLM only: the benchmark's standard agent, every turn is an LLM call | 0.807 | 1.00× | 0 / 0 / 100% |
| Jev only: every turn is a Jev pick, no LLM at all | 0.167 | 0.01× | 0 / 100% / 0 |
| Mixed Harness, before learning: the table holds only Jev's thresholds | 0.76 | 0.73× | 9% / 23% / 67% |
| Mixed Harness, after learning 42 rows from its own records | 0.746 | 0.57× | 29% / 23% / 49% |
| Mixed Harness, current: the same table, two gaps in the hand-written option list closed | 0.789 | 0.62× | 28% / 18% / 54% |

The current version solves 1.8 points fewer tasks than the LLM-only agent (within noise) at 0.62× its cost.
Of the two earlier harness rows, the first is the starting point and the second shows what learning alone
did: 10 points of cost at no change in tasks solved. Between the last two rows the code changed, not the
table: the hand-written list of options had never included the two address-change tools, so on those turns
Jev picked the nearest listed reply ("your request has been taken care of") with high probability, and a
closing reply was offered as soon as one of several requests was done. Before that fix the harness lost 17
tasks the LLM solved and solved 10 the LLM lost; after it, 11 and 9. Jev's own picks account for about 5% of
lost conversations. The rows in the learned table were learned from these same 114 tasks, so the harness
rows are in-sample numbers; the next experiment removes that.

**On conversations it has never seen.** The 114 tasks are split once into a stream of 76 tasks the table
learns from and 38 held-out tasks it never sees. Each pass serves the 76 stream tasks with a new simulated
customer, recomputes the table from everything served so far, and scores the held-out 38 with that table.
The LLM-only agent scores 0.921 on the held-out 38.

| conversations served | held-out pass^1 | held-out cost | held-out LLM share | learned rows |
|---|---|---|---|---|
| 76 | 0.842 | 0.75× | 69% | 0 |
| 152 | 0.842 | 0.66× | 57% | 7 |
| 228 | 0.868 | 0.62× | 54% | 18 |
| 304 | 0.816 | 0.60× | 50% | 30 |

Cost on unseen conversations falls at every pass; the LLM's share of steps falls from 69% to 50%; held-out
pass^1 stays within noise of the LLM's (the held-out set is 38 tasks, so its noise is wider than 9 points).
Two more passes are running; the flattening point of the curve is the open number. `scripts/long_run.py`
runs the whole protocol.

## Status

| Part | State |
|---|---|
| Runtime, sandbox, spec format; nodes skip on null input; malformed LLM replies are soft failures | done |
| Option listing for retail | done, hand-written (`examples/tau2/seeds.py`) |
| Option listing written by an LLM from schema and policy | not started |
| The table: code rows, split rows, Jev thresholds per state type, never rows | done, one script recomputes all of it from records (`scripts/fit_policy.py` with `--replay`, `--steps`, `--ledgers`) |
| Runtime path: code → Jev by probability → LLM → guard | done (`mixedharness/`, routed seed) |
| Guard: tool and argument checks, write gate on every path, fallback | done |
| Step table, ledgers, merge of partial runs | done (`mixedharness/trace.py`, `scripts/tau2_merge.py`) |
| Labeller and replay of Jev on recorded LLM-only turns | done (`scripts/stage0_label_replay.py`) |
| Learning curve on unseen conversations | running, 4 of 6 passes (`scripts/long_run.py`) |
| LLM roles on a ChatGPT subscription instead of an API | done (`scripts/cli_proxy.py`, `--llm-backend codex`) |
| Second domain | not started |

## Quick start

    uv sync --locked --extra dev
    uv run pytest                                   # mock providers, no keys needed
    uv run mixedharness doctor                      # sandbox and credentials present

Credentials go in a local `.env` (see `.env.example`): `TYPESAFE_API_KEY` for Jev (a `TYPESAFE_API_KEY_PERSONAL`
is used first when set), `ANTHROPIC_API_KEY` for the language model, or `--llm-backend codex` to run every
LLM role through a Codex CLI subscription behind `scripts/cli_proxy.py`. Sandboxed Python nodes run under
`bwrap` on Linux and `sandbox-exec` on macOS.

    ~/tau2-bench/.venv/bin/python scripts/stage0_label_replay.py                       # labels and Jev replay on recorded LLM-only traces
    ~/tau2-bench/.venv/bin/python scripts/fit_policy.py --replay runs/stage0/<stamp>/replay.jsonl --steps runs/tau2/<run>/routed/steps.jsonl
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --arms routed --policy examples/tau2/policy.json
    ~/tau2-bench/.venv/bin/python scripts/long_run.py --passes 6 --replay runs/stage0/<stamp>/replay.jsonl --llm-only-run runs/tau2/<llm_only run>

## Layout

| Path | What it holds |
|---|---|
| `mixedharness/` | the package: spec, expressions, sandbox, runtime, Jev and LLM clients, router (table fitting), trace, mining, seeds, evolution, freeze |
| `examples/tau2/` | the τ²-bench adapter, the agent wrapper, the routed seed and its two controls, the fitted table, a replay user |
| `scripts/` | `stage0_label_replay.py`, `fit_policy.py`, `learn_splits.py`, `tau2_compare.py`, `tau2_merge.py`, `learning_curve.py`, `long_run.py`, `cli_proxy.py` |
| `docs/design/` | `tau2-four-arms.md` (stages, predictions, every result), `implementation.md` |
| `docs/guides/` | `routed-harness.md` (reference), `tau2.md` (benchmark setup and how to read the table) |
| `docs/media/` | figures: `fig_runtime_path.svg`, `fig_evolution_loops.svg`, `fig_measurement.svg` |

The execution layer follows [JevHarness](https://github.com/TianyuCodings/JevHarness), a harness as a graph of
sandboxed code and Jev calls evolved with GEPA, implemented independently here; JevHarness has no LLM at
runtime and authors its harness once, offline. GEPA is Agrawal et al., arXiv:2507.19457. τ²-bench is Barres
et al., arXiv:2506.07982.
