#!/bin/bash
# One open-model judge on one GPU: serve it, replay the stage-0 questions on it, calibrate, and audit the LLM-only
# traces with it. Everything lands under runs/judges/<model tag>/.
#
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-4B scripts/slurm_judge_eval.sh
#
# Decision table (written 2026-10-13, before the first run): the calibrate step prints a verdict from its own kill
# test (>= 0.99 bin agreement >= 85% and share >= 10% of closed turns). "usable" -> fit a threshold table from the
# calibrated replay and run the routed arm on all 114 tasks with this judge (scripts/slurm_judge_server.sh keeps
# the model served for that run). "not usable" -> the next model up the ladder, not more prompt tuning.
#SBATCH -J judge-eval
#SBATCH -p gpu_requeue
# an 8B model in bf16 needs a whole A100 (a gpu_test MIG slice, 20 GB, went out of memory on 8B); 4B and smaller fit the slice
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /n/home12/shirleyhuang/mixed_harness/runs/judges/slurm-%j.out
set -u
cd /n/home12/shirleyhuang/mixed_harness
MODEL=${MODEL:-Qwen/Qwen3-4B}
TAG=$(echo "$MODEL" | tr '/' '-' )
NS=$(echo "$TAG" | tr -c 'A-Za-z0-9-\n' '-')   # cache namespaces are alphanumeric (Qwen3-1.7B has a dot)
OUT=runs/judges/$TAG
mkdir -p "$OUT"
JUDGE_PY=/n/home12/shirleyhuang/.local/opt/judge-venv/bin/python
TAU2_PY=/n/home12/shirleyhuang/tau2-bench/.venv/bin/python
PORT=$((8790 + RANDOM % 100))
export MIXEDHARNESS_SANDBOX=${MIXEDHARNESS_SANDBOX:-auto}
export HF_HUB_OFFLINE=1

# torch may still be installing on the login node when this job starts; wait up to 40 minutes for it
for i in $(seq 1 80); do
  $JUDGE_PY -c "import torch, transformers; torch.__version__; transformers.AutoModelForCausalLM" 2>/dev/null && break   # a half-copied install imports but has no version
  sleep 30
done
$JUDGE_PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || { echo "no torch"; exit 2; }
nvidia-smi -L

$JUDGE_PY scripts/judge_server.py --model "$MODEL" --port $PORT --host 127.0.0.1 > "$OUT/server.log" 2>&1 &
SERVER=$!
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && break
  sleep 5
done
curl -s "http://127.0.0.1:$PORT/v1/models" || { echo "server did not come up"; cat "$OUT/server.log"; kill $SERVER; exit 3; }
echo
JUDGE="logprob:$MODEL@http://127.0.0.1:$PORT/v1"
echo "judge: $JUDGE"

# 1. stage-0 replay on the same LLM-only traces the Jev replay used (runs/stage0/20261006-131412)
$TAU2_PY scripts/stage0_label_replay.py --results runs/tau2/20261001-stage2-v3a-merged/llm_only/results.json \
    --judge "$JUDGE" --out "$OUT/stage0" --namespace "judge-$NS-stage0" > "$OUT/stage0.log" 2>&1
rc=$?; echo "stage0 rc $rc"; tail -n 5 "$OUT/stage0.log"
[ $rc -eq 0 ] || { echo "stage0 failed; nothing downstream is read"; kill $SERVER; exit $rc; }
REPLAY=$(ls -d "$OUT"/stage0/*/replay.jsonl | tail -n 1)

# 2. calibrate: reliability table, temperature, kill test
$TAU2_PY scripts/calibrate_judge.py --replay "$REPLAY" --out "$OUT/judge.json" > "$OUT/calibrate.log" 2>&1
echo "calibrate rc $?"; cat "$OUT/calibrate.log"
TEMP=$($TAU2_PY -c "import json; print(json.load(open('$OUT/judge.json'))['temperature'])" 2>/dev/null || echo 1.0)

# 3. the audit of the gpt-5.5 LLM-only traces with this judge (plain and with the policy), for the README's ruler table
TRACE=runs/tau2/20261006-235933-retail-base/llm_only/results.json
$TAU2_PY -m mixedharness.cli audit $TRACE --format tau2 --judge "$JUDGE" --tools runs/audit/tools_retail.json --workers 4 --judge-temperature $TEMP \
    --namespace "judge-$NS-audit-plain" --out "$OUT/audit-plain" > "$OUT/audit-plain.log" 2>&1
echo "audit plain rc $?"
$TAU2_PY -m mixedharness.cli audit $TRACE --format tau2 --judge "$JUDGE" --tools runs/audit/tools_retail.json --policy-text runs/audit/policy_retail.md --workers 4 --judge-temperature $TEMP \
    --namespace "judge-$NS-audit-policy" --out "$OUT/audit-policy" > "$OUT/audit-policy.log" 2>&1
echo "audit policy rc $?"

# 4. the threshold table for this judge, from the calibrated replay
CAL="${REPLAY%.jsonl}.calibrated.jsonl"
$TAU2_PY scripts/fit_policy.py --replay "$CAL" --tau-only --judge-json "$OUT/judge.json" --out "$OUT/policy.json" > "$OUT/fit.log" 2>&1
echo "fit rc $?"; tail -n 3 "$OUT/fit.log"

kill $SERVER
echo "done: $OUT"
