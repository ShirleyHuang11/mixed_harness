#!/bin/bash
# Keep one open-model judge served on a GPU node for a routed run driven from elsewhere (the login node, where the
# Codex proxies and tau2_compare.py run). The server binds every interface; the node and port are written to
# runs/judges/<model tag>/server.addr as soon as it answers.
#
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-4B scripts/slurm_judge_server.sh
#   JUDGE="logprob:Qwen/Qwen3-4B@http://$(cat runs/judges/Qwen-Qwen3-4B/server.addr)/v1"
#SBATCH -J judge-serve
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /n/home12/shirleyhuang/mixed_harness/runs/judges/serve-%j.out
set -u
cd /n/home12/shirleyhuang/mixed_harness
MODEL=${MODEL:-Qwen/Qwen3-4B}
TAG=$(echo "$MODEL" | tr '/' '-')
OUT=runs/judges/$TAG
mkdir -p "$OUT"
JUDGE_PY=/n/home12/shirleyhuang/.local/opt/judge-venv/bin/python
PORT=$((8800 + RANDOM % 100))
export HF_HUB_OFFLINE=1
rm -f "$OUT/server.addr"
$JUDGE_PY scripts/judge_server.py --model "$MODEL" --port $PORT --host 0.0.0.0 > "$OUT/serve.log" 2>&1 &
SERVER=$!
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && break
  sleep 5
done
echo "$(hostname):$PORT" > "$OUT/server.addr"
echo "serving $MODEL at $(cat "$OUT/server.addr")"
wait $SERVER
