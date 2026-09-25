# Mixed Harness

An agent that serves customers, calls tools and follows a policy makes a decision on every turn. Most of
those decisions are small: which record to look up, which of four items the customer meant, whether they
just said yes. A standard LLM agent pays a full model call for each one, re-reading the policy, the tool
list and the whole conversation every time. On τ²-bench retail that is about 8,500 input tokens and 2
seconds per turn (measured on 2 tasks, see below). Some decisions are not small: an explanation the policy
does not script, a plan across several tool calls, a reply that has to be composed. For those, the full
model call is the right tool.

Mixed Harness starts from that difference. **Every step is routed: to Jev, TypeSafe's judgment model, when the
action space is closed; to a language model when it is open; and to a middle path where the language model
writes a short menu and Jev picks from it.** Which path a step takes is a decision the system makes, records,
and learns from, and the division of labor between Jev and the LLM moves over time as traces accumulate.

- **Closed path: Jev decides.** Code compiles the state into a typed action set, the facts that matter and a
  minimal state slice. Jev picks from that set in one request and returns calibrated probabilities. The
  pick is accepted when its confidence clears a per-cluster threshold; a small LLM call fills a free-text
  argument when one is needed. Cost: about one Jev call.
- **Narrow path: the LLM writes a menu, Jev picks.** For an open subproblem the LLM does not act; it
  compiles the open part into a few candidate actions and Jev chooses among them. Often the most useful
  mode in practice.
- **Escalate path: the LLM executes.** A full ReAct turn with tool calls, for open action spaces, low
  confidence, or a trajectory that has stalled.

A guard runs on every path: schema validation, precondition checks, and a hard escalation for irreversible
actions regardless of confidence. Every step is logged to a trace store with its route, action, outcome,
cost, Jev probabilities and harness version.

## Architecture

![Runtime path: task state, action-space compiler, Jev router, three paths, guard, trace store; the evolution loops feed a new harness version back into the execution layer and the router](docs/media/fig_runtime_path.svg)

**Why Jev shapes the design.** Jev returns only typed outputs, booleans, rubric scores and multiple-choice
picks, so Jev picks actions and free-text arguments need a small LLM slot filler. It is single-pass,
non-autoregressive and returns calibrated confidence, so an accept threshold is cheap to set and can be
recalibrated per domain with conformal prediction. Extraneous context lowers its accuracy and its context is
capped, so the compiler must produce a minimal state slice per question, and slice selection is something
to evolve. Its weights are proprietary, so the questions evolve, not the model. It is weaker on long horizons, so per-step confidence overstates trajectory
reliability and the router adds trajectory-level escalation.

**Runtime path.** The action-space compiler turns tool schemas and the current state into a typed action
set A(s), flags it as closed or open, and emits the minimal state slice for each Jev question. The Jev
router asks one choice question per state, closed, narrow or escalate, at the cost of about one Jev call.
Each path produces an action. The guard checks it, then it executes and is logged. Trajectory-level escalation
forces the LLM path after k consecutive low-margin steps or when progress stalls.

## The same step on the three paths

The customer writes: "I'm Yusuf Rossi, zip 19122, I want to exchange the keyboard in order #W2378156."
The compiler finds a name, a zip code and an order id and lists the actions that are legal right now; the
action set is closed (look the customer up, or ask for identification), so the router sends the step down
the closed path. Jev picks the lookup with confidence 0.72, above τ for this cluster, no argument is free
text, the guard accepts the call, and `find_user_id_by_name_zip(Yusuf, Rossi, 19122)` executes. Cost: one
Jev call.

Three turns later the customer asks for the specifications of the replacement items. The action set is
open, so the router does not send the step to Jev alone. On the narrow path the LLM writes a short menu
(quote the two variants, ask which matters more, offer to list all) and Jev picks one. On the escalate path
the LLM answers in a full turn with tool calls. Which of the two the router chooses is what loop B learns
from traces; which turns stop needing the LLM at all is what loop A's macro-action mining changes over time.

| | closed | narrow | escalate |
|---|---|---|---|
| who decides | Jev, from the compiled action set | Jev, from a menu the LLM wrote | the LLM |
| who writes | a small LLM slot filler, only for free-text arguments | the LLM, a few candidates | the LLM |
| cost per step | about one Jev call | one short LLM call and one Jev call | one full LLM call |
| guard | schema, preconditions, irreversible → escalate | same | same |

## Evolution loops

![Evolution loops: the trace store feeds the execution layer (slow) and the routing layer (fast); both deploy a new harness version](docs/media/fig_evolution_loops.svg)

The box at the bottom right of the runtime figure, opened up. Both loops read the trace store and both
deploy a new harness version; the slow one changes what Jev is asked, the fast one changes when Jev is
trusted. Each loop is two steps.

**Loop A, execution layer (slow).** Evolve the questions, not Jev's weights: question templates, option
sets and state-slice selection are programs, and GEPA optimizes them on traces with fitness = outcome reward
minus calibration error. Then mine macro-actions: tool-call subsequences that the LLM path repeats become new
options in Jev's choice set, so open territory becomes closed territory over time. The LLM explores, Jev
compiles and exploits. LLM prompts and tool docs evolve on the same traces; weights stay frozen on both sides.

**Loop B, routing layer (fast).** Recalibrate first: split conformal prediction sets τ(s) per state cluster
so the error rate on accepted Jev decisions stays below α, and is redone after every execution-layer change.
Then let a contextual bandit over the three paths move traffic, with reward = success minus λ·cost on
multi-step outcomes; sweeping λ traces the success-versus-cost frontier. With probability ε the bandit also
runs the path not taken and logs it, otherwise Jev only sees states the router already trusts it with and its
share cannot grow.

**Coupling.** Recalibrate before the bandit moves traffic, and warm-start the bandit after any
execution-layer change, or the router optimizes against a Jev that no longer exists. Every component is
version-pinned in the traces.

**Metrics that show the harness is evolving rather than tuning.**

- Jev traffic share over time.
- Escalation precision: how often the LLM disagreed with Jev's top choice on escalated states.
- Error rate on accepted Jev decisions, against α.
- The success versus cost and latency frontier.
- Trajectory-level completion rate by route mix.

## Measuring it

![The tau2-bench test: four arms on the same benchmark, one table](docs/media/fig_measurement.svg)

The claim is about the router, so the test scores each path alone and the router across them on the same
terms. On τ²-bench retail (114 tasks, 16 tools, reward computed from the database state) four arms run
every task against the same simulated user with the same seeds:

- **LLM only.** Every step escalates: the stock τ² agent, one full model call per turn.
- **Jev only.** Every step on the closed path: compiler, one Jev pick, code, templates for text. It exists to
  show where generation is needed. It is expected to lose.
- **Routed.** The router picks a path per step; the LLM is called only where Jev hands over.
- **Routed, evolved.** The same harness after loops A and B ran on the train split.

One table is generated from the artifacts: pass^1, dollars per task, milliseconds per turn, LLM and Jev
calls per turn, and the three evolution metrics above: Jev traffic share, error rate on accepted Jev
decisions against α, and escalation precision. The best single arm on each task is the per-task oracle.
The claim holds if the routed arm lands within noise of LLM only on pass^1 at a fraction of its cost; it
fails if the routed arm trails the best single arm. Predictions and the decision table are written down
before any run in `docs/design/tau2-three-arms.md`; `docs/guides/tau2.md` covers setup and reading.

Measured so far, on 2 retail tasks (the run stopped when the language-model account ran out of credit):
LLM only passed 2 of 2 at $0.17 per task and 2.1 s per turn; Jev only passed 0 of 2 at $0.004 per task and
0.47 s per turn, with one Jev request per turn and no LLM calls; the hand-written routed shape (closed path
plus a hand-to-LLM step) passed 1 of 2 with 0.33 LLM calls per turn. Two tasks say nothing about accuracy
(±35 points); they do say the instrument counts what it should, and that Jev answers in about 400 ms.

## Status

| Part | State |
|---|---|
| Runtime, sandbox, expression language, spec format | done, tested on mock providers |
| Closed path: compiler, Jev pick, code decision (the τ²-bench Jev-route seed) | done, run against real Jev |
| Escalate path: `agent` node, one native tool-calling turn | done |
| Guard: schema and argument checks, confirm-before-write in code | done in the seed's decision node; irreversible-action escalation not yet |
| Trace store: trajectories, rewards, route chosen vs realised | done (`mixedharness/store.py`, `triage_store`) |
| Loop A, question evolution (GEPA over the whole spec) | done, tested with a mock reflection model |
| Per-step Jev router with closed / narrow / escalate | partial: the hand-to-LLM option is a catalog entry Jev picks per step; a per-task dispatcher picks the seed |
| Narrow path (LLM menu, Jev picks) | not yet |
| Trajectory-level escalation | partial: the seed transfers after three identical replies |
| Loop A, macro-action mining | not yet |
| Loop B: conformal τ(s), router bandit with ε shadow runs | not yet |
| Routed arm in the benchmark table | partial: the runner scores LLM only, Jev only and the hand-written routed shape; the evolved arm takes a spec from a run |

## Quick start

    uv sync --locked --extra dev
    uv run pytest                                   # mock providers, no keys needed
    uv run mixedharness doctor                      # sandbox and credentials present

Credentials go in a local `.env` (see `.env.example`): `TYPESAFE_API_KEY` for Jev, `ANTHROPIC_API_KEY` for
the language model. Sandboxed Python nodes run under `bwrap` on Linux and `sandbox-exec` on macOS.

    uv run python scripts/live_run.py --task classify --rounds 2 --train 6 --validation 6
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --num-tasks 5

## Layout

| Path | What it holds |
|---|---|
| `mixedharness/` | the package: spec, expressions, sandbox, runtime, Jev and LLM clients, evolution, triage, routing, freeze |
| `examples/classify`, `examples/constrained_text` | a settled and an open-ended task that run end-to-end on mock providers |
| `examples/tau2/` | the τ²-bench adapter, the agent wrapper, and the route seeds |
| `scripts/` | `live_run.py` (small real-model routed run), `tau2_compare.py` (the four-arm benchmark test) |
| `docs/design/` | `routing.md`, `tau2-three-arms.md` (predictions and decision table), `implementation.md` |
| `docs/guides/` | `routed-harness.md` (reference), `tau2.md` (benchmark setup and how to read the table) |
| `docs/media/` | figures: `fig_runtime_path.svg`, `fig_evolution_loops.svg`, `fig_measurement.svg` (hand-drawn SVG) |

The execution layer follows the design of [JevHarness](https://github.com/TianyuCodings/JevHarness), a
harness as a graph of sandboxed code and Jev calls evolved with GEPA, implemented independently here. GEPA
is the optimizer of Agrawal et al., arXiv:2507.19457.
