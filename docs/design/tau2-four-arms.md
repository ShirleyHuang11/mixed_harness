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

Stage 0 result (2026-09-25, real Jev, `runs/stage0/20260925-172359/report.md`; 41 turns, 2 tasks each run twice):

| quantity | value |
|---|---|
| X, closed fraction by rule label | 80% [66%, 90%], n=41; tool turns 17 of 17 |
| X if every text label were wrong | 41% [28%, 57%] |
| positive control, identity-lookup turns | 4 of 4 |
| Jev = LLM on closed turns | 73% [56%, 85%], n=33; 85% counting greeting ≡ ask-identity |
| Jev = LLM at reported confidence ≥ 0.90 | 18 of 18 |
| Jev = LLM at reported confidence < 0.90 | 6 of 15 |
| paraphrase agreement | 32 of 33 agree; no resolution on this sample |
| Jev's own route pick on closed-label turns | 25 closed, 8 narrow |

The 9 misses are all below 0.55 confidence and none violates policy: 4 are the turn-1 greeting where Jev asks for
identity instead, 5 are ordering differences (load the profile before the order, or a write pick the guard would
have turned into a confirmation). The 8 open turns are the LLM composing variant lists with prices after a lookup;
the compiler calls them closed because a report template is legal, so the compiler over-claims on exactly the turns
where composition beats a template. Consequence for the router: a confidence threshold on the action pick is the
abstain signal that has resolution here, paraphrase agreement is not. With τ = 0.9 and Jev's route pick, 19 of 41
turns are accepted on the closed path, all 16 closed-label ones agree with the LLM, 3 open-label ones are accepted
(two would reach the guard's write gate). τ = 0.9 is now the seed default; per-cluster τ(s) is loop B's job once
there are enough steps per cluster.

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

Stage 2 result (2026-09-25, `runs/tau2/20260925-stage2-merged/table.md`, 114 tasks, all three arms complete, ±7.3 pt):

| arm | pass^1 | $/task | turns/task | Jev share | narrow | escalate | guard interventions |
|---|---|---|---|---|---|---|---|
| llm_only | 0.842 | 0.224 | 12.2 | 0 | 0 | 1.00 | 0 |
| jev_only | 0.158 | 0.0033 | 15.4 | 1.00 | 0 | 0 | 0 |
| routed | 0.439 | 0.271 | 21.7 | 0.26 | 0.56 | 0.18 | 93 |

**FAILS**: routed trails llm_only by 40.4 points and costs 1.2× as much. The positive control held (0.842 > 0.60) and
the instrument counted correctly (jev_only made no LLM calls). The failure is drift, not wrong picks: on the 66 tasks
with step records, 32 episodes contain three or more identical consecutive actions; the narrow path (Jev picked it on
57% of steps) produced `get_order_details` 190 times, twice in a row in 41 episodes and three times in 18; the write
gate fired 74 times in 41 episodes, 60 of them on the narrow path, because the menu's arguments differ slightly
from turn to turn and the gate compares them exactly, so a confirmed write is asked to be confirmed again. Lost
episodes have more narrow steps (60% vs 48%), more turns (23 vs 16) and more guard interventions (1.7 vs 1.0 per
episode) than won ones. Closed picks were accepted at confidence ≥ 0.9 on 329 of 344 steps. The routed arm's first 48
tasks lost their step records when the login node killed the run (a sandbox pool per agent leaked processes; fixed).

What this decides for the next version (loop A, by hand, from these records): (1) the write gate on the LLM paths
matches on tool and order, not on exact arguments; (2) a loaded order becomes the focused order, and the compiler asks
"which order" when several are loaded, so the catalog stops offering a lookup that already happened; (3) an identical
tool call repeated is "stuck" and escalates, and a lookup already in memory is answered from memory; (4) the narrow
path is off by default: a narrow pick escalates, since won episodes used escalate more and narrow produced a message
the LLM had written anyway on 410 of 746 steps. Version 2 is scored on the same 114 tasks against this table.

Routed version 2 (2026-10-01, `runs/tau2/20261001-stage2-v2-merged/table.md`; the five fixes above, narrow off):

| arm | pass^1 (110 common tasks) | $/task | turns/task | Jev share | LLM calls/turn | episodes with ≥3 identical replies |
|---|---|---|---|---|---|---|
| llm_only | 0.845 | 0.224 | 12.2 | 0 | 1.00 | 0% |
| jev_only | 0.164 | 0.0033 | 15.4 | 1.00 | 0 | — |
| routed v1 | 0.439 | 0.271 | 21.7 | 0.26 | 0.76 | 25% |
| routed v2 | 0.764 | 0.212 | 17.1 | 0.31 | 0.70 | 3% |

Four tasks (110 to 113) failed when the Anthropic credit ran out and are not in the table. The gap to llm_only went from
40 points to 8.1 (noise ±7.5), so the bugs were most of the loss; the remaining gap is at the edge of resolution. Per
task: 77 both pass, 16 only llm_only, 7 only routed, 10 both fail. Cost 0.95× llm_only, far from the pre-registered
0.5×: the LLM still takes 69% of steps (Jev's route pick plus 14% low-confidence escalations), each as expensive as an
llm_only turn, and v2 uses 40% more turns. The write gate fired 129 times in 78 episodes: the LLM path asks for
confirmation in its own words (a message, so no pending write is recorded), the customer says yes, the LLM calls the
write tool, and the gate, seeing no recorded pending write, asks again. Lost episodes carry more gates (1.35 vs 1.07
per episode) and more turns (18.2 vs 15.4). Verdict by the table: FAILS on cost, borderline on accuracy.

What this decides: the per-step router is not losing on wrong picks any more; it loses on turns. Version 3 is the
shared plan: the escalate path writes a short structured plan into memory and the closed path executes its steps,
so an LLM call happens per plan change rather than per open turn; and the LLM-path write gate accepts a yes that
follows the LLM's own confirmation message instead of demanding a recorded pending write.

Routed version 3a (2026-10-01, `runs/tau2/20261001-stage2-v3a-merged/table.md`): the route question is two-way
(closed / escalate) when narrow is off, an escalate pick needs route confidence ≥ 0.6, and a tool-result step with
listed options defaults to closed.

| arm | pass^1 (114) | $/task | turns/task | Jev share | LLM calls/episode | looping episodes |
|---|---|---|---|---|---|---|
| llm_only | 0.842 | 0.224 | 12.2 | 0 | 12.2 | 0% |
| routed v2 | 0.764 (110) | 0.212 | 17.1 | 0.31 | 11.2 | 3% |
| routed v3a | 0.807 | 0.208 | 17.9 | 0.37 | 10.6 | 3% |

Accuracy is now within noise of llm_only (3.5 points, ±7.3); per task 84 both pass, 12 only llm_only, 8 only routed,
10 both fail. Cost is 0.93×: INCONCLUSIVE ON COST by the table. The route question fix worked on its own terms, Jev
now picks closed on 71% of steps (was about 45%), but the action-confidence gate took over: 775 of 1,926 steps (40%)
escalated because the action pick's confidence was below τ = 0.9, median 0.73, 536 of them right after a tool
result, where several listed options are plausible (report the result, or fetch the next thing) and the probability
mass spreads. τ = 0.9 came from 41 turns in stage 0; the 114 LLM-only trajectories (about 1,400 turns) are the data
to recalibrate it, per cluster, which is loop B's conformal step with real support. Stickiness is unchanged: after an
escalate step the next step escalates 76% of the time.

Routed version 3b (2026-10-05, `runs/tau2/20261005-stage2-v3b-merged/table.md`): loop B deployed. Threshold per
cluster from the 1,389-turn replay (identity clusters 0.6; clusters right after a tool result 0.99 or never), path
preference per cluster from the v3a step table, both in `examples/tau2/policy.json`; the write gate accepts a yes
after the LLM's own confirmation question.

| arm | pass^1 (114) | $/task | turns/task | Jev share | LLM calls/episode | guard interventions |
|---|---|---|---|---|---|---|
| llm_only | 0.842 | 0.224 | 12.2 | 0 | 12.2 | 0 |
| routed v3a | 0.807 | 0.208 | 17.9 | 0.37 | 10.6 | 149 |
| routed v3b | 0.763 | 0.188 | 16.1 | 0.36 | 9.7 | 53 |

Cost moved to 0.84× and the gate fix removed two thirds of the guard interventions and two turns per episode, but
the Jev share did not move and pass^1 drifted down 4.4 points (within noise; per task 80 both, 12 only v3a, 7 only
v3b). The per-cluster thresholds did what they were fitted to do: of the 665 low-confidence escalations, 340 are in
the "never" clusters, all of the shape authenticated / tool result just arrived / no focused order. Those clusters
are a quarter of all steps in a routed run, and in the recorded LLM-only turns the LLM's own action there splits
three ways, load an order (30%), report the result (25%), ask which order (25%), while Jev picks "load the profile"
first. The 112 steps accepted below 0.9 under the relaxed thresholds sit in lost episodes at 37% against 24% overall,
so the 0.6 clusters are suspect too. Verdict: FAILS by 7.9 points, cost 0.84×.

What this decides: the remaining escalations are not a threshold problem, they are the workflow after a lookup, and
the LLM-only traces show that workflow is three deterministic rules (one order known → load it; several → ask which,
listing them; a result just arrived → report it). Version 4 puts those rules in the compiler as the step's default
when exactly one applies, asks Jev only when more than one does, and reverts the relaxed 0.6 thresholds to 0.9 until
the routed step tables, not the LLM-only replay, support them.

Routed version 4 (2026-10-06, `runs/tau2/20261006-stage2-v4-merged/table.md`): the structure settled in the grilling,
code first, then Jev's probability on its own pick, then the LLM, then the guard. The route and paraphrase questions
and the menu path are gone. Eight structural rules were written from the after-lookup analysis above and executed
without any model call when exactly one applied.

| arm | pass^1 (114) | $/task | turns/task | steps with no model call | Jev-decided | LLM steps |
|---|---|---|---|---|---|---|
| routed v3a | 0.807 | 0.208 | 17.9 | 0 | 37% | 63% |
| routed v4 | 0.658 | 0.144 | 19.3 | 39% | 25% | 36% |

Cost reached 0.64× and accuracy fell 15 points: FAILS. Replaying the eight rules on the 1,389 recorded LLM-only turns
gave each rule's agreement with the LLM: greeting 100%, single name and zip 97%, single email 100%, transfer notice
100%, load a mentioned unloaded order 8 of 9; but "report the result" 67%, "load the profile after authentication" 23%
(the LLM asks which order instead), "ask which order when several are known" 11% (the LLM loads them), "load the
product after the order" 30%. The four low-agreement rules fired on 757 of the 809 structural steps. The rule-writing
mistake was to read the after-lookup clusters as "mostly structural" when the LLM's action there splits three ways;
a rule is structural only when the traces agree with it, and the acceptance test for a rule is exactly that replay.
Jev-decided steps behaved as before: at confidence ≥ 0.9 they agree with the LLM on 99% (286 turns), below it 55%.

Version 4b keeps the five rules at ≥ 97% and returns the rest to Jev and the LLM.

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

## Learning curve under the Codex subscription (2026-10-06/07)

Every LLM role (user simulator, NL-assertion judge, the harness's fallback, the LLM-only agent) ran on gpt-5.5
through a ChatGPT subscription (`scripts/cli_proxy.py` for tau2's roles, `LLMClient(kind='codex')` for the
harness). Cost is imputed at gpt-5.5 list price ($5/$30 per MTok) for both arms, so the ratio is comparable;
the absolute numbers are not comparable with the Sonnet-simulator tables above. Run directories:
`runs/curve/20261006-180031` (tables and curve.jsonl), `runs/tau2/20261006-180046` (round 0),
`runs/tau2/20261006-curve-r1-merged` (round 1, 105 + 9 redone), `runs/tau2/20261006-231458` (round 2, partial),
`runs/tau2/20261006-231650` (jev_only, partial), `runs/tau2/20261006-235933` (llm_only).

| arm | table | tasks | pass^1 | $/task (imputed) | turns | code | Jev | LLM |
|---|---|---|---|---|---|---|---|---|
| llm_only | — | 114 | 0.807 | 0.351 | 12.2 | — | — | 100% |
| jev_only | — | 114 | 0.167 | 0.003 | 13.9 | — | 100% | — |
| routed, round 0 | seed (replay + v4b steps) | 114 | 0.763 | 0.255 (0.73×) | 13.5 | 9% | 23% | 67% |
| routed, round 1 | + round-0 steps | 114 | 0.728 | 0.249 (0.71×) | 13.3 | 11% | 23% | 66% |
| routed, round 2 | = round 1 (code rows and thresholds unchanged; only the unused `prefer` column differed) | 114 | 0.798 | 0.227 (0.65×) | 14.0 | 10% | 24% | 66% |
| routed, round 3 | + 42 split rows (policy_3) | 114 | 0.763 | 0.200 (0.57×) | 14.1 | 29% (19% learned) | 23% | 49% |

On the 77 tasks all three rounds scored: llm_only 0.792, round 0 0.740, round 1 0.740, round 2 0.831. Rounds 1
and 2 used the same table, so their gap is the noise of one 114-task pass under this simulator: 13 of 77 episodes
flipped, 9 points. On the 81 tasks jev_only scored: llm_only 0.802, jev_only 0.185.

What the table learned. After round 0, one code row (`auth|tool|exchange requested|exchange` → report the result,
Jev's accepted pick there 100% of the time and the replay's LLM action 97%) and three tightenings to never. After
round 1, nothing new: with 114 episodes a round, almost no state type reaches the n ≥ 10, ≥ 95% same-action bar.
The learning signal has to include the LLM fallback's own steps, or the state types have to be coarser, for the
table to move at this scale. Round 2 and jev_only stopped at 77/114 and 81/114 when TypeSafe credit ran out
(HTTP 402, 2026-10-07 01:30); `runs/curve/20261006-180031/RESUME.sh` finishes both.

Two instrument fixes came out of the run: a cluster now moves to code or tightens to never only when the stage-0
replay corroborates it (episode reward alone marked an identity cluster never after one lost episode), and the
Codex CLI occasionally fails to start under concurrency, so both the client and the proxy retry four times.

### Why the table did not move, and the fix: split rows

Across the four routed runs 2,884 steps went to the LLM. In none of the 40 state types did the LLM do the same
thing 90% of the time (top-action share 0.26 to 0.71), so no rule keyed on the state type alone can take those
steps, however many episodes are recorded. The state type is too coarse, not the sample too small. Adding one or
two cheap state features changes that: with `n_user_orders`, `n_loaded`, `last` (the tool whose result just
arrived), `products_loaded`, `last_user_yes` and `n_mentioned`, 30 subsets of at least 10 steps have one action at
>= 90% (`scripts/learn_splits.py`). Fitted on rounds 0 and 1 and tested on round 2's 590 LLM steps, those rows cover
30.5% of them and name the LLM's own action 91.7% of the time. The actions are three: load the profile after
authentication, load the next order from the profile, report the tool result that just arrived.

The compiler now computes those features (`state_features`, in the seed's common source, so the learner runs the
same function on ledgers) and consults `policy["splits"]` after its hand rules; a step a split row takes is
recorded with path source `learned`. `fit_policy.py --ledgers` learns the rows, and the curve passes each round's
ledgers to the next fit. Prediction for round 3 (table `policy_3.json`, 42 rows from all four ledgers covering
1,056 of 2,884 LLM steps): LLM share 0.62 to 0.67 → about 0.45, cost 0.66× → about 0.5× of llm_only, pass^1 within
the 9-point noise. The risk is on the "report the result" rows: the code path's templated report replaces the
LLM's prose, and 40 tasks are judged on the text.

### Round 3: the split rows moved the traffic

Round 3 ran the table with 42 split rows on all 114 tasks (2026-10-07). Prediction written before the run: LLM
share 0.62-0.67 → ~0.45, cost 0.66× → ~0.5×, pass^1 within the 9-point noise. Measured: LLM share 0.49, cost
0.57× ($0.200 against llm_only's $0.351), pass^1 0.763 (rounds 0-2: 0.728, 0.763, 0.798; llm_only 0.807). The
learned rows took 19.1% of all steps, two actions above all: report the tool result (104 steps) and load the
profile after authentication (78); the order-loading rows fired 20 times. Episodes with at least one learned step
passed at 0.827 (98 episodes), the 16 without at 0.625, so the rows are firing on the ordinary workflow, not on
the hard episodes. Jev's share did not change (0.23 in every round): the split rows take steps the LLM used to
take, which is the point.

Round 2 completed to 114 tasks (77 before the Jev outage + 37 after, same table): 0.798, $0.227 (0.65×). The
jev_only control completed to 114: 0.167.

One instrument incident: the round-2 resume and round 3 were launched in the same second and tau2_compare named
run directories by the second, so both wrote into `runs/tau2/20261007-133601-retail-base`. Round 3's summary,
steps and ledgers are its own in-memory state written at the end and are clean; round 2's 37 resumed results were
merged before being overwritten (verified by start stamps); round 2's step-level shares come from the 77-task run.
`results.json` in that directory mixes the two (tau2 keeps the first entry per task) and says so in a README.
Directories are now created exclusively.
