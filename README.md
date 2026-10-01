# Mixed Harness

An agent that serves customers, calls tools and follows a policy makes a decision on every turn. Most of
those decisions are small: which record to look up, which of four items the customer meant, whether they
just said yes. A standard LLM agent pays a full model call for each one, re-reading the policy, the tool
list and the whole conversation every time: about 8,300 input tokens and 2 seconds per turn on τ²-bench
retail. Some decisions are not small: an explanation the policy does not script, a plan across several tool
calls, a reply that has to be composed. For those, the full model call is the right tool.

Mixed Harness is a framework for splitting a tool-using task along that line and for learning where the
line is. **Code compiles each step into a typed action set, and Jev, TypeSafe's judgment model, picks from
it when it is confident. A language model takes the step when Jev is not confident. Every step is recorded,
and the next version of the compiler is generated from what the language model had to do.** The LLM
explores, the code takes over. The pieces are:

- **A compiler** that turns tool schemas, the policy and the current state into the legal actions, the
  arguments code can fill, and the questions a judgment model is asked. It is domain code. In the current
  version it is hand-written for one domain; the framework's claim is that it should be *generated*, by a
  language model, from the tool schemas, the policy, and what earlier runs recorded.
- **A router** that sends each step to the closed path (Jev decides, code fills the arguments, a template
  writes the text) or the escalate path (one native tool-calling turn of the LLM), from three signals: the
  compiler's closed flag, Jev's own route pick, and Jev's confidence on its action pick against a threshold
  calibrated per state cluster.
- **A guard** on every path: tool and argument checks, and a write gate that lets no database write through
  without the customer's yes to that operation.
- **A step table** that records, for every step, the cluster, the path and why, the confidence, the action,
  the guard's verdict, the cost and the episode's reward.
- **Two loops over the step table.** The slow one regenerates the compiler: it labels which steps the LLM
  took that code could have taken, mines the tool sequences the LLM repeats, and hands both to the generator
  as seeds. The fast one recalibrates the router: the confidence threshold per cluster, and the preferred path
  per cluster from outcomes.

## Architecture

![Runtime path: task state, compiler, Jev router, the paths, guard, trace store, and the loops feeding a new version back](docs/media/fig_runtime_path.svg)

**Why Jev shapes the design.** Jev returns only typed outputs, booleans, rubric scores and multiple-choice
picks, so Jev picks actions and code or an LLM fills free-text arguments. It is single-pass, non-autoregressive
and returns a probability with every pick, and that probability has resolution: on 1,002 recorded closed steps
it agreed with the LLM on 98% of picks at confidence 0.99 or above, 82% between 0.90 and 0.99, and 35% below
0.90. So a threshold is the abstain signal, and it is set from data per cluster. Extraneous context lowers its
accuracy and its context is capped, so the compiler must produce a minimal state slice per question. Its
weights are proprietary, so the questions evolve, not the model. It is weaker on long horizons. It picks
well when the options are a judgment and poorly when they are a plan, so the compiler carries the workflow
and the LLM takes the steps the compiler cannot express.

**Runtime path.** The execution layer is the compiler, the paths and the guard; the routing layer is the Jev
router between them. The compiler turns tool schemas and state into the legal action set, flags it closed
when every argument of every option can be filled from state and the customer's message maps to something
in it, and writes every question for one Jev request: the route pick, the action pick, a reordered paraphrase
of it, whether any option fits, whether the customer confirmed, and the candidate questions (which item, which
variant, which payment method, which order). The router applies the signals in order: compiler flag, Jev's
route pick, the confidence threshold on the action pick, the per-cluster policy, trajectory-level escalation
when an action repeats. The closed path executes; the escalate path hands the step to the LLM. The narrow
path, an LLM-written menu that Jev picks from, exists in the code and is off by default: on retail it cost
more than escalating and produced most of the loops. The guard checks the action, gates writes, and writes
the step record.

**The same step on the two paths.** The customer writes: "I'm Yusuf Rossi, zip 19122, I want to exchange the
keyboard in order #W2378156." The compiler finds a name, a zip code and an order id and lists the actions
that are legal right now, all fillable, so the step is closed. Jev picks the lookup with confidence 1.0, the
guard accepts the call, and `find_user_id_by_name_zip(Yusuf, Rossi, 19122)` executes. One Jev call. Three
turns later the customer asks which keyboard variants have clicky switches and what they cost. No template
composes that answer, Jev's confidence on the listed options is 0.6, and the step escalates: the LLM answers
in one tool-calling turn. The step record says the escalation happened after a product lookup, in a cluster
where Jev is unreliable, on a request the labeller calls open. Enough such records are the seed for a
compiler that has a "list variants with prices" template, and the step stops escalating.

## The loops

![Evolution loops: the trace store feeds the execution layer (slow) and the routing layer (fast); both deploy a new harness version](docs/media/fig_evolution_loops.svg)

**Loop A, the compiler is generated, not written.** The input is the step table and the LLM-only traces of
the task. Two miners read them: a labeller that marks each LLM step closed or open by what the LLM did (a
tool call whose arguments were all in the state, or a reply a template covers), and a sequence miner that
finds the tool-call runs the LLM repeats. Their output is the seed: which steps code should take over, in
what order, with what arguments. A language model then writes the compiler from the tool schemas, the policy
and that seed, and GEPA refines it on traces with strict improvement. The hand-written retail compiler is the
target this generator has to match; the test of the framework is a second domain with no hand-written code.

**Loop B, the router is calibrated, not designed.** The step table joined to outcomes gives, per cluster, the
agreement between Jev and the LLM at each confidence level, and the success rate of each path. The first sets
the threshold per cluster: on retail the identity clusters are reliable at any confidence, the clusters right
after a tool result need 0.99. The second is a bandit over the paths with reward success minus λ·cost, with
an ε share of shadow runs so a path the router distrusts still gets measured. Recalibration runs after every
compiler change, because a new compiler changes what Jev is asked.

**What JevHarness has and what this adds.** The execution layer follows
[JevHarness](https://github.com/TianyuCodings/JevHarness): a harness as a graph of sandboxed code and Jev
calls, one run per decision with memory across decisions, evolved with GEPA over complete traces. JevHarness
has no language model at runtime, so a step the harness does not cover is a wrong step, and its harness is
authored once, offline. Mixed Harness adds the runtime LLM path and the router, the guard, the step table,
and the loop that turns the LLM's steps into the next compiler. On τ²-bench retail the same compiler scores
0.158 without the LLM path and 0.807 with it.

## Measuring it

![The tau2-bench test: four arms on the same benchmark, one table](docs/media/fig_measurement.svg)

The claim is about the router and the loop, so the test scores each path alone and the router across them on
the same terms. On τ²-bench retail (114 tasks, 16 tools, reward computed from the database state) every arm
runs every task against the same simulated user with the same seeds: LLM only (the benchmark's own agent),
Jev only (every step on the closed path), routed, and routed after a loop. One table is generated from the
artifacts: pass^1, dollars per task, turns per task, calls per turn, Jev traffic share, and the step table's
routing metrics. Before any routed run, the labeller replays Jev on the recorded LLM-only turns, so the
closed fraction and Jev's reliability per cluster are measured first. The decision rules, predictions and
every result so far are in `docs/design/tau2-four-arms.md`; `docs/guides/tau2.md` covers setup.

Measured so far, on all 114 retail tasks (noise ±7.3 points on pass^1):

| arm | pass^1 | $ per task | turns per task | Jev share of steps |
|---|---|---|---|---|
| LLM only | 0.842 | 0.224 | 12.2 | 0 |
| Jev only | 0.158 | 0.003 | 15.4 | 1.00 |
| routed, first version | 0.439 | 0.271 | 21.7 | 0.26 |
| routed, third version | 0.807 | 0.208 | 17.9 | 0.37 |

The first routed version looped: a quarter of its episodes repeated the same reply three or more times. The
fixes came from its step table (a looked-up order becomes the focus, a repeated lookup is answered from memory,
the write gate matches the operation rather than its exact arguments, a repeated action escalates, the narrow
path is off, the route question is two-way) and brought accuracy within noise of the LLM at 0.93× its cost.
The cost half of the claim is open: Jev still takes only 37% of steps, because 40% of steps escalate on
confidence below 0.9 and the data says those escalations are right. Those steps are the "which lookup next"
decisions after a tool result, which are a plan, not a judgment; on the 1,389 recorded LLM-only turns Jev
agrees with the LLM on 98% of identity-phase picks and on 43% to 56% in the clusters after a tool result.
That is the gap the generated compiler has to close, by carrying the workflow so that Jev's pick is a judgment
again.

## Status

| Part | State |
|---|---|
| Runtime, sandbox, expression language, spec format; nodes skip on null input; soft failure for malformed LLM replies | done, tested on mock providers |
| Compiler for τ²-bench retail: catalog, argument fill, closed flag, cluster, questions | done, hand-written (`examples/tau2/seeds.py`) |
| Compiler generator: LLM writes the compiler from schemas, policy and mined seeds | designed, not built |
| One Jev request per step with route pick, action pick, paraphrase, fits, candidates | done |
| Router: compiler flag → Jev pick → confidence threshold → policy with ε shadow → trajectory escalation | done; τ = 0.9 global, per-cluster values measured but not yet deployed |
| Closed and escalate paths; narrow path (off by default) | done |
| Guard: tool and argument checks, write gate on every path, fallback | done |
| Step table with routing record and episode reward; merge of partial runs | done (`mixedharness/trace.py`, `scripts/tau2_merge.py`) |
| Loop A, labeller and sequence miner | done (`scripts/stage0_label_replay.py`, `mixedharness/mining.py`) |
| Loop A, GEPA over the whole spec | done, tested with a mock reflection model |
| Loop B, per-cluster policy from steps embedded in the spec | done (`mixedharness/router.py`); per-cluster τ not yet wired |
| Four-arm benchmark runner, replay user for credit-free plumbing | done; results above |
| Second domain with no hand-written compiler | not started; the test of generality |

## Quick start

    uv sync --locked --extra dev
    uv run pytest                                   # mock providers, no keys needed
    uv run mixedharness doctor                      # sandbox and credentials present

Credentials go in a local `.env` (see `.env.example`): `TYPESAFE_API_KEY` for Jev, `ANTHROPIC_API_KEY` for
the language model. Sandboxed Python nodes run under `bwrap` on Linux and `sandbox-exec` on macOS.

    ~/tau2-bench/.venv/bin/python scripts/stage0_label_replay.py                       # labels and Jev replay on recorded LLM-only traces
    ~/tau2-bench/.venv/bin/python scripts/tau2_compare.py --domain retail --split base --num-tasks 5
    uv run python scripts/live_run.py --task classify --rounds 2 --train 6 --validation 6

## Layout

| Path | What it holds |
|---|---|
| `mixedharness/` | the package: spec, expressions, sandbox, runtime, Jev and LLM clients, router, trace, mining, seeds, evolution, freeze |
| `examples/classify`, `examples/constrained_text` | a settled and an open-ended task that run end-to-end on mock providers |
| `examples/tau2/` | the τ²-bench adapter, the agent wrapper, the routed seed and its two controls, a replay user |
| `scripts/` | `stage0_label_replay.py` (labels and Jev replay), `tau2_compare.py` (the four-arm benchmark), `tau2_merge.py`, `live_run.py` |
| `docs/design/` | `tau2-four-arms.md` (stages, predictions, every result), `implementation.md` |
| `docs/guides/` | `routed-harness.md` (reference), `tau2.md` (benchmark setup and how to read the table) |
| `docs/media/` | figures: `fig_runtime_path.svg`, `fig_evolution_loops.svg`, `fig_measurement.svg` |

GEPA is the optimizer of Agrawal et al., arXiv:2507.19457. τ²-bench is Barres et al., arXiv:2506.07982.
