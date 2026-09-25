# τ²-bench, four arms: the testing plan

Date: 2026-09-25 (supersedes the three-arm design of 2026-09-24). Runner: `scripts/tau2_compare.py`; stage 0:
`scripts/stage0_label_replay.py`; seeds: `examples/tau2/seeds.py`.

## The claim (claim A)

On τ²-bench retail, a measured fraction X of steps is closed: the legal action set is enumerable from state and
every argument of the action taken can be filled from state. On those steps a judgment model (Jev) picks what the
LLM would have picked, at about one Jev call per step instead of one full LLM call, and a per-step router sends the
rest to the LLM. The routed harness lands within noise of the LLM-only agent on pass^1 at a fraction of its cost.

Honesty line: the compiler for retail is hand-written domain code (about 500 lines). The contribution is the
routing and the learning on top of it; generality is stage 3's question, not this plan's.

## Stages, each with a kill number

**Stage 0. Label and replay what is on disk. No LLM credit.**
Ground truth is what the LLM did on the LLM-only trajectories already recorded (4 tasks, 41 assistant turns):
a tool call is closed iff its option id is in the compiled catalog and every argument value appears in the state
at that point; a text reply is closed iff it matches a template intent that is legal at that point. The compiler's
closed flag is scored against the label. Then the Jev request (route, action, reordered paraphrase, fits) is
replayed on every turn with real Jev. Outputs: X with a Wilson interval, the compiler-flag confusion matrix, Jev's
agreement with the LLM on closed turns, and the same by reported confidence and by paraphrase agreement, which
decides whether a confidence threshold has any resolution or agreement is the better abstain signal.
- Positive control: Jev = LLM on identity-lookup turns ≥ 90%, else the slice or the question is broken.
- Prediction: X = 60 to 70%. Kill: X < 30%, or Jev more than 10 points under the LLM on closed turns.
- The rule labeller's text labels are written with `needs_review`; the report gives X both ways.

**Stage 1. Scale the labels. About $30 of LLM credit.**
LLM-only on all 114 retail tasks (about $0.21 per task for the agent plus $0.03 for the user simulator), relabel,
X with ±5 points, replay Jev on every closed turn. This number is what the README claim rests on.

**Stage 2. The four-arm run. About $80.**
`llm_only`, `jev_only`, `routed`, `routed_evolved` on 114 tasks, same seeds and simulator. Resolution at n=114 is
±7.5 points on pass^1. Written before running: llm_only ≥ 0.60 (positive control, else the batch is void);
routed within 7.5 points of llm_only at ≤ 0.5× its agent cost; jev_only well below both.
- Holds: routed within h of llm_only and cost ≤ 0.5×. Fails: routed trails the best single arm by more than h,
  or costs more than 0.5× llm_only while matching it (reported as inconclusive on cost).
- `routed_evolved` takes a spec from an evolution run on the train split (74 tasks) and is scored on the test
  split (40), interval ±12 points; or the routed seed with a loop-B policy fitted on the `routed` arm's steps.

**Stage 3, only if stage 2 holds: a second domain.** Airline (50 tasks, 14 tools), compiler written fresh and
timed; and τ³ `banking_knowledge` for the first genuinely open tasks, where the narrow path is tested on something
a template cannot cover.

## Arms

| Arm | Agent | Per turn |
|---|---|---|
| `llm_only` | tau2's stock `llm_agent` on `--agent-llm` | 1 LLM call with policy, tools and full history |
| `jev_only` | `jev_only_seed()`: compile → judge → route → decision → guard, every step forced closed | 1 Jev request, no LLM; Jev's route pick recorded |
| `routed` | `routed_seed()`: the same plus menu → menu_q → menu_pick and agent | 1 Jev request, plus one short LLM call and one Jev pick on the narrow path, or one full LLM turn on the escalate path |
| `routed_evolved` | a candidate from evolution, or the routed seed with a fitted loop-B policy | as routed |

All see the same policy text and tool schemas. The step table `<arm>/steps.jsonl` records, per turn: cluster,
compiler flag, Jev's route pick, the realised path and why (`jev`, `code_override`, `fallthrough`, `policy`,
`shadow`, `trajectory`, `forced`, `guard`), agreement, confidence, action, guard outcome, cost, latency, reward.

## Preemption check (ten minutes, non-blocking)

- τ²-bench, arXiv:2506.07982: LLM agents only; no judgment model, no code-first agent.
- TypeSafe workflow evals (evals.typesafe.ai): Jev "workflow" vs LLM "prompt" on four private tasks; labels are
  the consensus of two LLMs. Ours is a public benchmark with DB-state truth.
- Select-then-Solve, arXiv:2604.06753; Meta-Harness, arXiv:2603.28052: route or rewrite whole reasoning
  strategies; neither splits a single turn into a typed judgment and a generation.
- 2026 frontier models with tools sit at 98 to 99% on retail (artificialanalysis.ai/evaluations/tau2-bench). The
  claim is not SOTA accuracy; it is matched accuracy against the same LLM used alone, at a fraction of the cost
  and latency per turn: an oracle-vs-practice pair where both curves are ours to measure.

## Domain

Retail, `base` split, 114 tasks, 16 tools, reward = DB state plus natural-language assertions (40 tasks carry 61)
or DB state alone. 36 tasks require a value to be communicated (a refund amount, a tracking number, a count of
variants), every one of which comes out of a tool result, so the content is closed where the wording is free.
Two tasks have no expected tool action. Telecom is excluded: its reward is on the user's device state.

Official user simulator: `gpt-4.1-2025-04-14`, temperature 0, unchanged across arms. This machine's OpenAI key
has no credit, so the runs so far used `anthropic/claude-sonnet-4-5-20250929` for the simulator and the NL judge;
the manifest records both. Numbers from different simulators are not comparable.

## Costs

LLM-only ≈ $0.21 per task for the agent (measured on 2 tasks: 8,457 input tokens per turn, 11 turns) plus about
$0.03 for the simulator. Jev ≈ $0.004 per task. A full four-arm stage 2 is about $80; stage 1 about $30.
