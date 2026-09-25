# Mixed Harness

**Code computes the facts, a judgment model makes the bounded choices, and a language model writes only the
text that has to be written.** Mixed Harness builds such a decision pipeline from a task description, then
improves it on two levels: the pipeline itself, through GEPA evolution over its full execution traces, and
the triage that chose the pipeline's shape, through replay against recorded outcomes.

The execution layer follows [JevHarness](https://github.com/TianyuCodings/JevHarness): a harness is a graph
of sandboxed Python nodes and calls to [Jev](https://docs.typesafe.ai), TypeSafe's judgment model, which
answers typed questions with probabilities in a few hundred milliseconds. Mixed Harness adds the routing
layer, `llm` nodes for generation, and the second improvement loop.

## Architecture

![Mixed Harness architecture](docs/media/fig1_architecture.svg)

**Triage.** A task description (objective, action space, evaluator, a few examples) goes through a free,
deterministic probe that computes nine fields: the declared action kind, the label count, whether the
examples carry a legal-action catalog, whether the evaluator is exact, the shape of the expected output.
Jev reads those fields and the examples and returns a probability over three routes. Each route has a
seed template:

- **settled**: the decision picks from a finite set. Seed: code → Jev pick → code.
- **open-ended**: the output is free text that code can check. Seed: code → LLM draft → code.
- **hybrid**: a few candidates are generated and Jev picks among them. Seed: LLM propose → code criteria → Jev select.

One harness is seeded from the chosen template.

**Loop 1: the harness improves.** The seeded harness runs on training tasks and every run leaves a complete
trace: node inputs and outputs, Jev questions and answers, LLM prompts, memory, failures, and the score.
GEPA's reflection model reads those traces and proposes a rewritten graph. The proposal is not limited to
prompt text: it can add or remove nodes, change a node's kind, insert a generate-then-select structure, or
move logic from a model call into Python. A proposal is kept only if it beats its parent on the same
training batch and holds up on validation. Parents are drawn from a Pareto frontier over task instances, so
the search does not collapse onto one lineage. A wrong triage is corrected here rather than by re-running
the triage: the summary of every run records both the route the triage chose and the route the selected
harness turned out to have.

**Loop 2: the triage improves.** That pair, triage route versus final route, is appended to a history after
every run. The reflection model proposes a new triage question map; the old and the new map are both
replayed through Jev over the whole history, and the new one is accepted only if it agrees with the recorded
outcomes strictly more often. Accepted maps are versioned.

## Inside one decision

![One decision inside a harness](docs/media/fig2_one_decision.svg)

The same three node kinds run in every harness. A **python** node parses the input, keeps track of what
earlier tool calls returned, and builds the list of legal actions together with every question the turn
might need. One **jev** node asks all of those questions in a single request; adding questions barely
changes its latency. A second python node turns the answers into a tool call or a template message and
enforces the rules that code can enforce, such as confirmation before any write. An **llm** node exists for
the cases no template covers. When code decides the model is not needed, the node's prompt is `null` and
the runtime skips it: no call, no cost. The retail seed in `examples/tau2/seeds.py` is exactly this graph.

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
