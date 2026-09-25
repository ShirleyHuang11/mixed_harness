# Mixed Harness

An agent that serves customers, calls tools and follows a policy makes a decision on every turn. Most of
those decisions are small: which record to look up, which of four items the customer meant, whether they
just said yes. A standard LLM agent pays a full model call for each one, re-reading the policy, the tool
list and the whole conversation every time. On τ²-bench retail that is about 8,500 input tokens and 2
seconds per turn (measured on 2 tasks, see below). Some decisions are not small: an explanation the policy
does not script, a plan across several tool calls, a reply that has to be composed. For those, the full
model call is the right tool.

Mixed Harness starts from that difference. **A probe decides what kind of task this is, and the task is
dispatched to a route of execution built for that kind.** One route is driven by a judgment model, one by
a language model, one by both. Which route runs is a decision the system makes, records, and learns from.

- **The Jev route.** Code computes the facts (ids in the message, what earlier tool calls returned, which
  actions are legal right now, the rules that must never depend on a model). [Jev](https://docs.typesafe.ai),
  TypeSafe's judgment model, takes those facts and typed questions (pick one of these, yes or no, rate on a
  scale) and answers all of them in one request with probabilities, in a few hundred milliseconds. Code
  turns the answers into the action. No language model is called.
- **The LLM route.** A complete language-model agent: the policy, the tool schemas and the conversation go to
  the model with native tool calling, and the model decides and writes. Tasks that need generation or
  multi-step reasoning go here. The route is the same thing a standard agent framework runs.
- **The mixed route.** The Jev route's code and judgment, plus a language model that is called only on the
  turns where something has to be written or extracted that code cannot. On every other turn the model is
  skipped: no call, no cost.

A task that needs no model at all (sort these words, count these objects) goes to a fourth, code-only route.

## Architecture

![Mixed Harness architecture](docs/media/fig1_architecture.svg)

There is one kind of machine here, used at two levels. A **route** decides what to do on every turn of a
task. The **triage** decides which route a task gets, once per task. Both are the same three-node shape
(code, one Jev request, code), both run through the same runtime, and both improve by the same rule:
propose a change, replay it, keep it only if the score rises.

**Triage.** The triage takes the task description (objective, action space, evaluator, a few examples) as
its observation. Its python node is a free, deterministic probe that computes nine fields: the declared
action kind, the label count, whether the examples carry a legal-action catalog, whether the evaluator is
exact, the shape of the expected output. Its Jev node reads those fields and the examples and returns a
probability over the routes. Code then fills the winning route's template from the task description; that
filled template is the seed the route starts from. The decision is recorded with its full distribution, so
a low-confidence dispatch is visible and can be overridden.

**Two levels of routing.** The triage routes once per task. Inside the mixed route, the same question is
asked again on every turn, cheaply: the Jev request that picks the action also decides whether this turn
needs the language model. Most of the saving comes from that second level, because in a tool-using task
most turns are lookups and confirmations.

**Loop 1: the routes improve.** The seeded route runs on training tasks and every run leaves a complete
trace: node inputs and outputs, Jev questions and answers, LLM prompts and tool calls, memory, failures, and
the score. An evolutionary loop reads those traces, proposes a rewritten route, and keeps the proposal only
if it beats its parent on the same training batch and holds up on validation. For the Jev and mixed routes
a proposal can add or remove nodes, change a node's kind, insert a generate-then-select structure, or move
logic from a model call into code. For the LLM route it rewrites the agent's instructions. The loop is
defined by that acceptance rule, not by one optimizer: GEPA is the current implementation (its reflection
model writes the proposals and its Pareto frontier over task instances picks the parents), and any
proposer that respects the same rule can replace it. A wrong dispatch is corrected here rather than by
re-running the triage: the summary of every run records both the route the triage chose and the route the
evolved graph turned out to be.

**Loop 2: the triage improves.** That pair, route chosen versus what the graph became, is appended to a
history after every run. The reflection model proposes a new question map for the triage's Jev node; the
old and the new map are both replayed over the whole history, and the new one is kept only if it agrees
with the recorded outcomes strictly more often. It is loop 1 with a different training set: past tasks
instead of past turns.

## What the mixed route does on one turn

The architecture figure builds and improves a route. This figure is one route opened up: what it does every
time it has to act, once per customer message or tool result.

![One turn of the mixed route](docs/media/fig2_one_decision.svg)

Walk one turn of the retail seed (`examples/tau2/seeds.py`). The customer writes: "I'm Yusuf Rossi, zip
19122, I want to exchange the keyboard in order #W2378156."

1. **facts (python).** Code reads the conversation and finds a name, a zip code and an order id. It checks
   what earlier tool calls already returned (nothing yet) and lists the actions that are legal right now:
   look the customer up by name and zip, or ask for identification. It also writes down every question the
   judge might need this turn.
2. **judge (Jev).** All of those questions go to Jev in one request. Jev answers each with a probability:
   the action should be the name-and-zip lookup (0.72); the customer has not confirmed anything yet (0.05).
   More questions in the same request cost almost no extra time.
3. **decision (python).** Code turns the answers into a concrete action: the tool call
   `find_user_id_by_name_zip(Yusuf, Rossi, 19122)`. Rules that must not depend on a model live here too.
   One example: no database write happens before the customer has seen the details and said "yes".
4. **draft (LLM), only when needed.** This turn needs no free text, so the decision node sets the draft
   prompt to `null` and the runtime skips the node: no call, no cost. When a customer asks something no
   template covers, the prompt is a string and the LLM writes the reply.

The action goes to the environment; what the turn learned (ids seen, records fetched) is carried as memory
into the next turn's observation. The Jev route is this graph without step 4. The LLM route is step 4 alone,
with the tools handed to the model.

## Measuring it

![The tau2-bench test](docs/media/fig3_three_arms.svg)

The claim is about the dispatch, so the test scores the routes and the dispatcher on the same terms. On
τ²-bench retail (114 tasks, 16 tools, reward computed from the database state), every route runs every task
against the same simulated user with the same seeds: the Jev route, the LLM route, the mixed route. The
best of the three on each task is the oracle. The fourth row is the triage's own pick. One table is
generated from the artifacts: pass^1, dollars per task, milliseconds per turn, and LLM and Jev calls per
turn, for each route, for the oracle, and for the dispatcher. The claim holds if the dispatcher lands within
noise of the oracle's accuracy at a fraction of the LLM route's cost; it fails if the dispatcher trails the
best single route. Predictions and the decision table are written down before any run in
`docs/design/tau2-three-arms.md`; `docs/guides/tau2.md` covers setup and reading.

Measured so far, on 2 retail tasks (the run stopped when the language-model account ran out of credit):
the LLM route passed 2 of 2 at $0.17 per task and 2.1 s per turn; the Jev route passed 0 of 2 at $0.004 per
task and 0.47 s per turn, with one Jev request per turn and no LLM calls; the mixed route passed 1 of 2
with 0.33 LLM calls per turn. Two tasks say nothing about accuracy (±35 points); they do say the instrument
counts what it should, and that Jev answers in about 400 ms.

## Status

| Part | State |
|---|---|
| Runtime, sandbox, expression language, spec format | done, tested on mock providers |
| Jev route and mixed route seeds for τ²-bench retail | done, run against real Jev |
| Triage as a spec through the same runtime, loop 2 | done |
| GEPA evolution of a route (loop 1) | done, tested with a mock reflection model |
| LLM route with native tool calling | not yet: today's `llm` node returns schema-checked JSON only |
| Code-only route | not yet |
| Dispatcher row and oracle in the benchmark table | not yet: the runner scores the three routes as arms |

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
| `scripts/` | `live_run.py` (small real-model routed run), `tau2_compare.py` (the benchmark test) |
| `docs/design/` | `routing.md`, `tau2-three-arms.md` (predictions and decision table), `implementation.md` |
| `docs/guides/` | `routed-harness.md` (reference), `tau2.md` (benchmark setup and how to read the table) |
| `docs/media/` | figures and the JSON specs they are rendered from |

The execution layer follows the design of [JevHarness](https://github.com/TianyuCodings/JevHarness), a
harness as a graph of sandboxed code and Jev calls evolved with GEPA, implemented independently here. GEPA
is the optimizer of Agrawal et al., arXiv:2507.19457. Figures are rendered from the specs in
`docs/media/specs/`; edit the spec, not the SVG.
