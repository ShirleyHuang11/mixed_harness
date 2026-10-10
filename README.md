# Mixed Harness

<p align="center"><b>A harness for tool-using agents: code, a 400 ms judgment model and an LLM share the turns, and the split is learned from the harness's own records.</b></p>

<p align="center">
<img alt="benchmark" src="https://img.shields.io/badge/benchmark-%CF%84%C2%B2--bench%20retail-4a4a9c">
<img alt="cost" src="https://img.shields.io/badge/cost-0.75%C3%97%20of%20an%20LLM%20agent-0f6e56">
<img alt="accuracy" src="https://img.shields.io/badge/accuracy-within%20noise%20of%20the%20LLM-0f6e56">
<img alt="tests" src="https://img.shields.io/badge/tests-78%20passing-534ab7">
</p>

---

## ⚡ In one minute

- **What it is.** A harness, not an agent: the layer around the models that lists what a turn allows, decides
  who answers it (code, Jev or the LLM), checks the answer, records it, and learns the split. Put around a
  task's tools and policy, the harness with Jev and an LLM is the agent that gets measured.
- **The problem.** A customer-service agent decides something on every turn. Most decisions are small (which
  record, which item, did they say yes). An LLM agent pays a full model call for each one: ~8,300 tokens and
  2 seconds per turn on τ²-bench retail.
- **The idea.** Code lists what the current state allows. **Jev**, TypeSafe's judgment model (400 ms, a
  probability with every answer), picks among the listed actions. The pick is used when its probability
  clears a threshold for that type of state; otherwise the **LLM** takes the turn. Every turn is recorded.
- **The learning.** After every batch of conversations the thresholds are refit from where Jev's picks
  matched the LLM's. No hand-tuned router: a table of thresholds, read off the records.
- **The result.** On τ²-bench retail: tasks solved within noise of the LLM-only agent at about **0.75× its
  cost**; on conversations it has never seen, cost settles near 0.75× after 150 conversations. A variant that
  also learns code rows reaches **0.62×** and keeps falling.

---

## 🎁 What it gives you

Measured on τ²-bench retail (114 customer-service tasks, the same simulated customer for every system; every
number is explained in the sections below and recorded in `docs/design/tau2-four-arms.md`).

| Advantage | What was measured | Number |
|---|---|---|
| **Cheaper than an LLM agent, same tasks solved** | cost per task relative to the benchmark's own LLM agent, and tasks solved | main harness: about 0.75× the cost, tasks solved within noise of the LLM; the code-row variant: 0.62× at 0.789 against the LLM's 0.807 |
| **It gets cheaper the more it is used, on conversations it has never seen** | 76 tasks served per pass, the table refit after each pass, 38 never-seen tasks scored each time | main harness: 0.80× → 0.75× over the first 152 conversations, then flat; the code-row variant: 0.75× → 0.60× over 304 conversations, accuracy unchanged |
| **Nothing in the routing is hand-tuned** | where the thresholds come from | every threshold is the lowest value at which Jev's recorded picks matched the LLM's at least 90% of the time; state types where no threshold reaches that are marked never |
| **Every decision is inspectable** | what a turn leaves behind | the state type, who took the turn and why, Jev's probability, the action, and later the episode's outcome; the table is a short list a person can read |
| **The LLM stays in the loop for what needs it** | which turns the LLM still takes | open requests, plans across several tool calls, composed replies; about 60% of turns on retail for the main harness, 50% for the variant |
| **Jev is cheap and fast where it is trusted** | Jev's share and its agreement with the LLM | 40% of turns; on accepted picks 92% to 98% agreement, 0.4 s per turn, about one hundredth of an LLM call |

> Not yet shown: accuracy parity with a confidence interval (one seed per point, 9 points of noise on 114
> tasks), a second domain, and cost on a real bill rather than list-price tokens.

---

## 🧭 How it works

![Runtime path: task state, compiler, Jev router, the paths, guard, trace store, and the loops feeding a new version back](docs/media/fig_runtime_path.svg)

**At runtime, every turn goes through five stations:**

1. **Compiler (code).** Reads the conversation, remembers what it has fetched, keeps the customer's request
   list (asked for, minus done), and lists the actions the state allows: tools whose arguments can be filled,
   known entities (orders, items, payment methods), templated replies.
2. **Jev router (one request).** Jev answers three things at once: is this turn *closed* (one listed action is
   right) or does it go to the LLM; which listed action; and the candidate questions (which order, which
   variant, did the customer confirm).
3. **Who takes the turn.** Jev, when it said closed and its probability on the pick is at or above the
   threshold τ for this type of state. The LLM, otherwise, in one native tool-calling turn. A turn whose
   listed actions cannot cover the request goes to the LLM without asking. (The optional menu path, where the
   LLM writes a short menu and Jev picks, is off: it cost more than it saved.)
4. **Guard.** Checks tool names and arguments, lets no database write through without the customer's yes to
   that operation, falls back to the listed default when a model's output is unusable.
5. **Trace store.** The turn is recorded: state type, who took it and why, Jev's probability, the action, and
   later the episode's outcome.

![Evolution loops: the trace store feeds the execution layer (slow) and the routing layer (fast); both deploy a new harness version](docs/media/fig_evolution_loops.svg)

**After every batch of conversations, two loops read the trace store:**

- **Fast loop: recalibrate τ per state type.** For every turn the LLM took, the record holds Jev's own pick and
  probability next to what the LLM did. A state type gets the lowest τ at which Jev's picks at or above τ
  agreed with the LLM at least 90% of the time; where no τ reaches that, the type is marked *never*.
- **Slow loop: regenerate the compiler** from the tool schema, the policy document and the patterns in the
  records. In the current version the compiler is hand-written for retail, and the patterns the records
  surfaced (the request list, two tools missing from the catalog) were applied by hand.

**Why Jev shapes the design.** Jev returns typed outputs only, so code or an LLM fills free text. Its
probability is calibrated: on 1,002 recorded closed turns it agreed with the LLM on 98% of picks at 0.99 or
above, 82% between 0.90 and 0.99, and 35% below 0.90, so the probability is what the threshold reads. Its
route answer separates closed from open turns (85% of its "closed" calls are closed) but does not say whether
its pick is right (52% on those turns), which is why the threshold sits on the pick, not on the route answer.
It picks well when the options are a judgment and poorly when the next turn depends on a plan the
conversation implies (44% in the state types right after a lookup), so those types are marked never until the
records show otherwise.

---

## 📋 The table on retail

One threshold per state type, computed from 1,389 recorded LLM-only turns and refit from every batch of
routed conversations since. A state type is: the authentication state, whether a customer message or a tool
result just arrived, the status of the order in focus, and the intent. A sample:

| state type | τ | basis |
|---|---|---|
| not authenticated, customer message | 0.6 | Jev agrees with the LLM on 98% of picks at ≥ 0.6, n = 168 |
| authenticated, order delivered, return intent | 0.6 | 100% at ≥ 0.6, n = 19 |
| authenticated, a lookup just returned, no order in focus | never | 52% to 71% even at ≥ 0.9 |
| authenticated, order pending, cancel intent, customer message | 0.6 | after refit from 304 routed conversations |
| everything else | 0.9 | 92% at ≥ 0.9 |

After 304 routed conversations the table holds 38 thresholds, 18 of them never. Refitting moves turns between
Jev and the LLM; it does not move turns to code, which is the difference between this harness and the
code-row variant at the end of the results.

**What Jev is for.** The turns where the actions are listed and the arguments are known but a judgment
remains: which record to load next, which variant the customer means, which payment method, whether the
reply is a yes. About 40% of all turns; the LLM takes about 60% and nearly all of the cost. Jev is not trusted
on its own say-so: its route answer is one signal, its probability on the pick is the one that gates.

---

## 📊 Results

![The tau2-bench test: four arms on the same benchmark, one table](docs/media/fig_measurement.svg)

The benchmark is **τ²-bench retail**: 114 customer-service conversations (returns, exchanges, cancellations,
address changes) with 16 tools and a simulated customer. A task counts as solved when the database ends in
the right state and the required things were said. Every system below runs the same 114 tasks against the
same simulated customer (gpt-5.5) with the same seed. Decision rules, predictions and every result:
`docs/design/tau2-four-arms.md`.

**How to read the tables**

- **tasks solved**: the fraction of the 114 tasks solved (pass^1). Running the same system twice moves this
  by up to 9 points, so smaller differences are noise.
- **cost**: what the agent spent on model calls, relative to the LLM-only agent (1.00×). Jev costs about one
  hundredth of an LLM call, so this column is almost entirely LLM calls.
- **who took the turns**: the share of turns decided by code (no model call), by Jev, and by the LLM.

| system | tasks solved | cost | who took the turns: code / Jev / LLM |
|---|---|---|---|
| **LLM only**: the benchmark's standard agent, every turn is an LLM call | 0.807 | 1.00× | 0 / 0 / 100% |
| **Jev only**: every turn is a Jev pick, no LLM at all | 0.167 | 0.01× | 0 / 100% / 0 |
| **Mixed Harness**: Jev routes and picks, the LLM takes the rest | run in progress under this simulator; the held-out table below is the measurement so far | | |

### 🔁 On conversations it has never seen

The 114 tasks are split once into a **stream** of 76 tasks the table learns from and **38 held-out tasks it
never sees**. Each pass serves the 76 stream tasks with a new simulated customer, refits the thresholds from
everything served so far, and scores the held-out 38 with that table. The LLM-only agent scores 0.921 on the
held-out 38 (noise on 38 tasks is wider than 9 points).

| conversations served | held-out tasks solved | held-out cost | held-out turns: Jev / LLM | thresholds (never) |
|---|---|---|---|---|
| 76 | 0.816 | 0.80× | 37% / 63% | 21 (9) |
| 152 | 0.816 | 0.75× | 43% / 57% | 29 (15) |
| 228 | 0.842 | 0.73× | 41% / 59% | 35 (18) |
| 304 | 0.895 | 0.76× | 40% / 60% | 38 (18) |

Accuracy stays within noise of the LLM's; the first refit buys 5 cost points and the curve then settles,
because refitting thresholds only moves turns between Jev and the LLM. `scripts/long_run.py --tau-only` runs
the protocol.

### 🧪 A variant that keeps lowering cost

`examples/tau2/seeds_splits.py` adds a second kind of table row: a state type, plus one or two cheap state
features, that the records show the LLM always handles the same way becomes a **code row** and is taken with no
model call. Under the same protocol its held-out cost falls **0.75× → 0.66× → 0.62× → 0.60×** at the same
accuracy, with 25% of turns on code by the last pass; on all 114 tasks it solves 0.789 at 0.62×. It is the
better number and the more complex harness; the figures above describe the main one.

---

## 🧩 Status

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

## 🔌 Install in Claude Code

Inside Claude Code:

```text
/plugin marketplace add https://github.com/ShirleyHuang11/mixed_harness.git
/plugin install mixed-harness@mixedharness
/reload-plugins
```

Then describe your task (the tools, the policy, how an episode is scored) and invoke the `mixed-harness`
skill: it establishes the task contract, writes the option lister for your tools from the retail one, runs
stage 0 and the three arms, and fits the thresholds. The skill is `skills/mixed-harness/SKILL.md`; it works
the same way as a plain document for Codex or any other coding agent.

## 🔑 Requirements

| You need | For | Notes |
|---|---|---|
| **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/) | the package | `uv sync --locked --extra dev` installs everything; `uv run pytest` runs on mock providers with no keys |
| **A sandbox** | the code nodes | `bwrap` (bubblewrap) on Linux, `sandbox-exec` on macOS; `uv run mixedharness doctor` tells you which it found |
| **`TYPESAFE_API_KEY`** | Jev | from [console.typesafe.ai](https://console.typesafe.ai/keys); direct API, about $0.04 per million tokens, ~400 ms per request. A `TYPESAFE_API_KEY_PERSONAL` is used first when set. Needed by the Jev-only arm, the harness, and stage 0 |
| **An LLM**, one of: | the LLM-only arm, the harness's LLM turns, the simulated customer and the judge | |
| &nbsp;&nbsp;`ANTHROPIC_API_KEY` | | the default API backend (`claude-sonnet-5`); a 114-task run of one arm is roughly $25 to $40 |
| &nbsp;&nbsp;a Codex CLI subscription | | `--llm-backend codex`: every LLM role goes through `scripts/cli_proxy.py` (start it first); no key, no bill; about 2 hours per 114-task arm |
| `OPENAI_API_KEY` (optional) | tau2's official simulated customer (`--user-llm gpt-4.1-2025-04-14`) | without it the simulator runs on the LLM above; numbers from different simulators are not comparable |
| **[tau2-bench](https://github.com/sierra-research/tau2-bench)** | the benchmark | Python 3.12 in its own venv with this repo installed into it (`docs/guides/tau2.md`); only for measuring, not for using the harness on your own task |

Keys go in a local `.env` next to `.env.example`; `.env` and `.env.*` are ignored by git and never read from
anywhere else. Runs write to `runs/` (also ignored): results, ledgers, step tables, the generated table. There
is no telemetry; the only network calls are to the providers you configured.

## 🚀 Quick start
## 🚀 Quick start

    uv sync --locked --extra dev
    uv run pytest                                   # mock providers, no keys needed
    uv run mixedharness doctor                      # sandbox and credentials present

Credentials and the sandbox: see Requirements above.

    ~/tau2-bench/.venv/bin/python scripts/stage0_label_replay.py                       # labels and Jev replay on recorded LLM-only traces
    ~/tau2-bench/.venv/bin/python scripts/fit_policy.py --tau-only --replay runs/stage0/<stamp>/replay.jsonl --ledger-replay runs/tau2/<run>/routed/ledgers.json
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --arms routed --policy examples/tau2/policy.json
    ~/tau2-bench/.venv/bin/python scripts/long_run.py --passes 4 --tau-only --replay runs/stage0/<stamp>/replay.jsonl --llm-only-run runs/tau2/<llm_only run>

## 🗂 Layout

| Path | What it holds |
|---|---|
| `mixedharness/` | the package: spec, expressions, sandbox, runtime, Jev and LLM clients, router (table fitting), trace, mining, seeds, evolution, freeze |
| `examples/tau2/` | the τ²-bench adapter, the wrapper that presents the harness to the benchmark as its agent, the routed seed (`seeds.py`) and its two controls, the fitted thresholds (`policy.json`), the split-row variant (`seeds_splits.py`), a replay user |
| `scripts/` | `stage0_label_replay.py`, `fit_policy.py`, `learn_splits.py`, `tau2_compare.py`, `tau2_merge.py`, `learning_curve.py`, `long_run.py`, `cli_proxy.py` |
| `docs/design/` | `tau2-four-arms.md` (stages, predictions, every result), `implementation.md` |
| `docs/guides/` | `routed-harness.md` (reference), `tau2.md` (benchmark setup and how to read the table) |
| `docs/media/` | figures: `fig_runtime_path.svg`, `fig_evolution_loops.svg`, `fig_measurement.svg` |
| `skills/mixed-harness/` | the Claude Code skill: how to build and measure a harness for a new task, with an integration reference |
| `.claude-plugin/` | plugin and marketplace manifests for `/plugin install` |

The execution layer follows [JevHarness](https://github.com/TianyuCodings/JevHarness), a harness as a graph of
sandboxed code and Jev calls evolved with GEPA, implemented independently here; JevHarness has no LLM at
runtime and authors its harness once, offline. GEPA is Agrawal et al., arXiv:2507.19457. τ²-bench is Barres
et al., arXiv:2506.07982.
