# Mixed Harness

An agent that serves customers, calls tools and follows a policy makes a decision on every turn. Most of
those decisions are small: which record to look up, which of four items the customer meant, whether they
just said yes. A standard LLM agent pays a full model call for each one, re-reading the policy, the tool
list and the whole conversation every time. On τ²-bench retail that is about 8,500 input tokens and 2.4
seconds per turn for the stock agent (measured on 2 tasks, see below).

Mixed Harness splits every decision into three kinds of work and gives each to the cheapest thing that can
do it:

- **Code computes the facts.** The ids in the message, what earlier tool calls returned, which actions are
  legal right now, and the rules that must never depend on a model, such as "no write before the customer
  says yes".
- **A judgment model makes the bounded choices.** [Jev](https://docs.typesafe.ai), TypeSafe's judgment
  model, takes a state and typed questions (pick one of these, yes or no, rate on a scale) and answers all
  of them in one request with probabilities, in a few hundred milliseconds.
- **A language model writes only what has to be written.** Free text for the customer, or an argument no
  code can extract. On a turn where nothing needs writing it is not called at all.

A harness turn is therefore one Jev request plus a few milliseconds of code, and a language-model call only
on the turns that need one. Two more things follow from the split. The rules live in code, so they hold on
every turn instead of being a hope about a prompt. And the harness is a small data file, a graph of nodes,
so the system can build it from a task description and keep rewriting it from its own execution traces:
the graph improves through GEPA evolution, and the triage that picks the graph's shape improves by replay
against what past harnesses turned out to need.

The execution layer follows the design of [JevHarness](https://github.com/TianyuCodings/JevHarness), a
harness as a graph of sandboxed code and Jev calls evolved with GEPA, implemented independently here. Mixed
Harness adds the triage, `llm` nodes for generation, and the second improvement loop.

## Architecture

![Mixed Harness architecture](docs/media/fig1_architecture.svg)

There is one kind of machine here, used at two levels. A **task harness** decides what to do on every
turn of a task. A **triage harness** decides which task harness to build, once per task. Both are the same
three-node shape (code, one Jev request, code), both run through the same runtime, and both improve by the
same rule: propose a change, replay it, keep it only if the score rises.

**Triage.** The triage harness takes the task description (objective, action space, evaluator, a few
examples) as its observation. Its python node is a free, deterministic probe that computes nine fields: the
declared action kind, the label count, whether the examples carry a legal-action catalog, whether the
evaluator is exact, the shape of the expected output. Its Jev node reads those fields and the examples and
returns a probability over three routes. Its decision node picks the seed template for the winning route:

- **settled**: the decision picks from a finite set. Seed: code → Jev pick → code.
- **open-ended**: the output is free text that code can check. Seed: code → LLM draft → code.
- **hybrid**: a few candidates are generated and Jev picks among them. Seed: LLM propose → code criteria → Jev select.

One harness is seeded from the chosen template.

**Loop 1: the task harness improves.** The seeded harness runs on training tasks and every run leaves a
complete trace: node inputs and outputs, Jev questions and answers, LLM prompts, memory, failures, and the
score. GEPA's reflection model reads those traces and proposes a rewritten graph. The proposal is not
limited to prompt text: it can add or remove nodes, change a node's kind, insert a generate-then-select
structure, or move logic from a model call into Python. A proposal is kept only if it beats its parent on
the same training batch and holds up on validation. Parents are drawn from a Pareto frontier over task
instances, so the search does not collapse onto one lineage. A wrong triage is corrected here rather than by
re-running the triage: the summary of every run records both the route the triage chose and the route the
selected harness turned out to have.

**Loop 2: the triage harness improves.** That pair, route chosen versus what the harness became, is
appended to a history after every run. The reflection model proposes a new question map for the triage
harness's Jev node; the old and the new map are both replayed over the whole history, and the new one is
kept only if it agrees with the recorded outcomes strictly more often. It is the same rule as loop 1 with a
different training set: past tasks instead of past turns.

## The execution layer: what the green box does on one turn

The architecture figure builds and improves a harness. This figure is the green box from that figure,
opened up: what the harness does every time it has to act, once per customer message or tool result.

![The execution layer: one turn](docs/media/fig2_one_decision.svg)

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
into the next turn's observation. Every harness the system builds is a graph of these same three node
kinds, and loop 1 in the architecture figure is GEPA rewriting exactly this graph.

## Measuring it

![The tau2-bench test](docs/media/fig3_three_arms.svg)

The claim is that a harness reaches the accuracy of the same language model used alone at a fraction of the
cost and latency per decision. The test runs three agents on τ²-bench retail (114 tasks, 16 tools, reward
computed from the database state) against the same simulated user with the same seeds: the stock LLM
agent, a Jev-only harness with no language model at all, and the full harness. One table is generated from
the artifacts: pass^1, dollars per task, milliseconds per turn, and LLM and Jev calls per turn. The
predictions and the decision table are written down before any run in
`docs/design/tau2-three-arms.md`; `docs/guides/tau2.md` covers setup and reading.

## Quick start

    uv sync --locked --extra dev
    uv run pytest                                   # mock providers, no keys needed
    uv run mixedharness doctor                      # sandbox and credentials present

Credentials go in a local `.env` (see `.env.example`): `TYPESAFE_API_KEY` for Jev, `ANTHROPIC_API_KEY` for llm
nodes. Sandboxed Python nodes run under `bwrap` on Linux and `sandbox-exec`
on macOS.

    uv run python scripts/live_run.py --task classify --rounds 2 --train 6 --validation 6
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --num-tasks 5

## Layout

| Path | What it holds |
|---|---|
| `mixedharness/` | the package: spec, expressions, sandbox, runtime, Jev and LLM clients, evolution, triage, routing, freeze |
| `examples/classify`, `examples/constrained_text` | a settled and an open-ended task that run end-to-end on mock providers |
| `examples/tau2/` | the τ²-bench adapter, the agent wrapper, and the two seed harnesses |
| `scripts/` | `live_run.py` (small real-model routed run), `tau2_compare.py` (the three-arm test) |
| `docs/design/` | `routing.md` (the routed harness), `tau2-three-arms.md` (the benchmark test, with predictions and decision table) |
| `docs/guides/` | `routed-harness.md` (reference), `tau2.md` (benchmark setup and how to read the table) |
| `docs/media/` | figures and the JSON specs they are rendered from |

Figures are rendered from the specs in `docs/media/specs/`; edit the spec, not the SVG. GEPA is the
optimizer of Agrawal et al., arXiv:2507.19457.
