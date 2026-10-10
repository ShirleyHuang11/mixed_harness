# Mixed Harness

An agent that serves customers, calls tools and follows a policy makes a decision on every turn. Most of
those decisions are small: which record to look up, which of four items the customer meant, whether they
just said yes. A standard LLM agent pays a full model call for each one, re-reading the policy, the tool
list and the whole conversation every time: about 8,300 input tokens and 2 seconds per turn on τ²-bench
retail. Some decisions are not small: an explanation the policy does not script, a plan across several tool
calls, a reply that has to be composed. For those, the full model call is the right tool.

Mixed Harness splits a tool-using task along that line and learns where the line is from its own records.
**Code lists the actions the current state allows. Jev (TypeSafe's judgment model: 400 ms, typed output, a
probability with every answer) is asked two things in one request: whether this step is closed (one of the
listed actions is the right one) or has to go to the LLM, and which listed action it is. Jev's pick is used
when its probability clears the threshold the table holds for that type of state; otherwise the LLM takes the
step. Every step is recorded, and after every batch of conversations the thresholds are refit from the record
of where Jev's picks matched the LLM's.** On τ²-bench retail the harness runs within noise of the LLM-only
agent's accuracy at about three quarters of its cost, and on conversations it has never seen its cost settles
near 0.75× of the LLM's after 300 conversations.

## What it gives you

Measured on τ²-bench retail (114 customer-service tasks, the same simulated customer for every system; details
and every number in the sections below and in `docs/design/tau2-four-arms.md`).

| Advantage | What was measured | Number |
|---|---|---|
| **Cheaper than an LLM agent, same tasks solved** | cost per task relative to the benchmark's own LLM agent, and tasks solved | main harness: about 0.75× the cost, tasks solved within noise of the LLM (the 114-task row is in the table below); the code-row variant: 0.62× at 0.789 against the LLM's 0.807 |
| **It gets cheaper the more it is used, on conversations it has never seen** | 76 tasks served per pass, the table refit after each pass, 38 never-seen tasks scored each time | main harness: 0.80× → 0.75× over the first 152 conversations, then flat; the code-row variant: 0.75× → 0.60× over 304 conversations, accuracy unchanged |
| **Nothing in the routing is hand-tuned** | where the thresholds come from | every threshold is the lowest value at which Jev's recorded picks matched the LLM's at least 90% of the time; state types where no threshold reaches that are marked never |
| **Every decision is inspectable** | what a step leaves behind | the state type, who took the step and why, Jev's probability, the action, and later the episode's outcome; the table is a short list a person can read |
| **The LLM stays in the loop for what needs it** | which steps the LLM still takes | open requests, plans across several tool calls, composed replies; about 60% of steps on retail for the main harness, 50% for the variant |
| **Jev is cheap and fast where it is trusted** | Jev's share and its agreement with the LLM | 40% of steps; on accepted picks 92% to 98% agreement, 0.4 s per step, about one hundredth of an LLM call |

What this does not yet show: accuracy parity with a confidence interval (one seed per point, 9 points of
noise on 114 tasks), a second domain, and cost on a real bill rather than list-price tokens.

## Architecture

![Runtime path: task state, compiler, Jev router, the paths, guard, trace store, and the loops feeding a new version back](docs/media/fig_runtime_path.svg)

**Runtime, every step.** The compiler (code) parses the conversation, keeps the entities it has fetched in
memory, keeps the customer's request list (the operations asked for, minus the ones already done), and lists
the actions the state allows: the tools whose arguments can be filled, the known entities (orders, items,
payment methods), the templated replies. The Jev router is one request with the route question (closed or
escalate), the action pick, and the candidate questions (which order, which variant, did the customer
confirm). Jev decides the step when it picked closed and its probability on the action is at or above the
threshold τ for this type of state; a step whose listed actions cannot cover the request goes to the LLM
without asking. The LLM executes the step otherwise, in one native tool-calling turn. The optional menu path
(the LLM writes a short menu, Jev picks) is off: it cost more than it saved. The guard checks tool names and
arguments, lets no database write through without the customer's yes to that operation, and falls back to
the listed default when a model's output is unusable. The step goes to the trace store with the state type,
who took it, the probability, the action and later the episode's reward.

![Evolution loops: the trace store feeds the execution layer (slow) and the routing layer (fast); both deploy a new harness version](docs/media/fig_evolution_loops.svg)

**Evolution, after every batch of conversations.** Two loops read the trace store. The fast one recalibrates
τ per state type: for every step the LLM took, the record holds Jev's own pick and probability next to what
the LLM did, so each batch adds agreement observations; a state type gets the lowest τ at which Jev's picks at
or above τ agreed with the LLM at least 90% of the time, and a state type where no τ reaches that is marked
never (always the LLM). The slow one regenerates the compiler from the tool schema, the policy document and
the patterns in the records; in the current version the compiler is hand-written for retail, and the patterns
found in the records were applied by hand (the request list, two tools that were missing from the catalog).

**Why Jev shapes the design.** Jev returns typed outputs only, so code or an LLM fills free text. Its
probability is calibrated: on 1,002 recorded closed steps it agreed with the LLM on 98% of picks at 0.99 or
above, 82% between 0.90 and 0.99, and 35% below 0.90, so the probability is what the threshold reads. Its
route answer separates closed from open steps (85% of its "closed" calls are closed) but does not by itself
say whether its pick is right (52% on those steps), which is why the threshold sits on the action pick and not
on the route answer. It picks well when the options are a judgment and poorly when the next step depends on a
plan the conversation implies (44% in the state types right after a lookup), so those state types are marked
never until the records show otherwise.

## The table on retail

The table is one threshold per state type, computed from 1,389 recorded LLM-only turns and refit from every
batch of routed conversations since. A state type is the authentication state, whether a customer message or
a tool result just arrived, the status of the order in focus, and the intent. A sample:

| state type | τ | basis |
|---|---|---|
| not authenticated, customer message | 0.6 | Jev agrees with the LLM on 98% of picks at ≥ 0.6, n = 168 |
| authenticated, order delivered, return intent | 0.6 | 100% at ≥ 0.6, n = 19 |
| authenticated, a lookup just returned, no order in focus | never | 52% to 71% even at ≥ 0.9 |
| authenticated, order pending, cancel intent, customer message | 0.6 | after refit from 304 routed conversations |
| everything else | 0.9 | 92% at ≥ 0.9 |

After 304 routed conversations the table holds 38 thresholds, 18 of them never. Refitting moves steps
between Jev and the LLM; it does not move steps to code, which is the difference between this harness and the
split-row variant below.

**What Jev is for.** Jev decides the steps where the actions are listed and the arguments are known but a
judgment remains: which record to load next, which variant the customer means, which payment method, whether
the reply is a yes. It takes about 40% of all steps; the LLM takes about 60% and all of the cost. Jev is not
trusted on its own say-so: its route answer is one signal, its probability on the pick is the one that gates.

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
| Mixed Harness: Jev routes and picks, the LLM takes the rest | run in progress under this simulator; the held-out table below is the measurement so far | | |

**On conversations it has never seen.** The 114 tasks are split once into a stream of 76 tasks the table
learns from and 38 held-out tasks it never sees. Each pass serves the 76 stream tasks with a new simulated
customer, refits the thresholds from everything served so far, and scores the held-out 38 with that table.
The LLM-only agent scores 0.921 on the held-out 38 (noise on 38 tasks is wider than 9 points).

| conversations served | held-out tasks solved | held-out cost | held-out steps: Jev / LLM | thresholds (never) |
|---|---|---|---|---|
| 76 | 0.816 | 0.80× | 37% / 63% | 21 (9) |
| 152 | 0.816 | 0.75× | 43% / 57% | 29 (15) |
| 228 | 0.842 | 0.73× | 41% / 59% | 35 (18) |
| 304 | 0.895 | 0.76× | 40% / 60% | 38 (18) |

Accuracy stays within noise of the LLM's; the first refit buys 5 cost points and the curve then settles,
because refitting thresholds only moves steps between Jev and the LLM. `scripts/long_run.py --tau-only` runs
the protocol.

**A variant that keeps lowering cost.** `examples/tau2/seeds_splits.py` adds a second kind of table row: a
state type, plus one or two cheap state features, that the records show the LLM always handles the same way
becomes a code row and is taken with no model call. Under the same protocol its held-out cost falls 0.75× →
0.66× → 0.62× → 0.60× at the same accuracy, with 25% of steps on code by the last pass; on all 114 tasks it
solves 0.789 at 0.62×. It is the better number and the more complex harness; the figures above describe the
main one.

## Status

| Part | State |
|---|---|
| Runtime, sandbox, spec format; nodes skip on null input; malformed LLM replies are soft failures | done |
| Option listing for retail | done, hand-written (`examples/tau2/seeds.py`); the request list and the full write-tool catalog came from the records |
| Option listing written by an LLM from schema and policy | not started |
| The table: one threshold per state type, never rows; refit from the replay and from every recorded batch | done (`scripts/fit_policy.py --tau-only --ledger-replay`, `examples/tau2/policy.json`) |
| Runtime path: route question → Jev pick by probability → LLM → guard | done (`mixedharness/`, routed seed) |
| Guard: tool and argument checks, write gate on every path, fallback | done |
| Step table, ledgers, merge of partial runs | done (`mixedharness/trace.py`, `scripts/tau2_merge.py`) |
| Labeller and replay of Jev on recorded LLM-only turns | done (`scripts/stage0_label_replay.py`) |
| Learning curve on unseen conversations | done, 4 passes (`scripts/long_run.py --tau-only`) |
| The code-row variant (`seeds_splits.py`, `scripts/learn_splits.py`) | done, measured; not the main harness |
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
    ~/tau2-bench/.venv/bin/python scripts/fit_policy.py --tau-only --replay runs/stage0/<stamp>/replay.jsonl --ledger-replay runs/tau2/<run>/routed/ledgers.json
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --arms routed --policy examples/tau2/policy.json
    ~/tau2-bench/.venv/bin/python scripts/long_run.py --passes 4 --tau-only --replay runs/stage0/<stamp>/replay.jsonl --llm-only-run runs/tau2/<llm_only run>

## Layout

| Path | What it holds |
|---|---|
| `mixedharness/` | the package: spec, expressions, sandbox, runtime, Jev and LLM clients, router (table fitting), trace, mining, seeds, evolution, freeze |
| `examples/tau2/` | the τ²-bench adapter, the agent wrapper, the routed seed (`seeds.py`) and its two controls, the fitted thresholds (`policy.json`), the split-row variant (`seeds_splits.py`), a replay user |
| `scripts/` | `stage0_label_replay.py`, `fit_policy.py`, `learn_splits.py`, `tau2_compare.py`, `tau2_merge.py`, `learning_curve.py`, `long_run.py`, `cli_proxy.py` |
| `docs/design/` | `tau2-four-arms.md` (stages, predictions, every result), `implementation.md` |
| `docs/guides/` | `routed-harness.md` (reference), `tau2.md` (benchmark setup and how to read the table) |
| `docs/media/` | figures: `fig_runtime_path.svg`, `fig_evolution_loops.svg`, `fig_measurement.svg` |

The execution layer follows [JevHarness](https://github.com/TianyuCodings/JevHarness), a harness as a graph of
sandboxed code and Jev calls evolved with GEPA, implemented independently here; JevHarness has no LLM at
runtime and authors its harness once, offline. GEPA is Agrawal et al., arXiv:2507.19457. τ²-bench is Barres
et al., arXiv:2506.07982.
