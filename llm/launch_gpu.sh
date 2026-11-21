#!/bin/bash
# Run one GPU job per lab machine in tmux.
# usage (from 981_project/, on SFU Wi-Fi or VPN):
#   llm/launch_gpu.sh <tag> <jobs.txt> host1 ... hostN
# jobs.txt has one shell command per line (run from the code folder with the
# environment active); line i runs on host i. Commands should write their
# outputs under $OUT, which is ~/rlvr_results/llm/<tag>/<host> (network home,
# survives the 03:00 /tmp wipe). Large files (model weights, envs, HF cache)
# stay in /tmp/asa715. Each host builds the environment first if it is missing
# (llm/setup_env.sh), then writes $OUT/DONE (or $OUT/FAILED) at the end.
set -e
TAG=$1; JOBS=$2; shift 2; HOSTS=("$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT=$(git -C "$ROOT" rev-parse --short HEAD)
N=$(wc -l < "$JOBS")
[ "$N" -le "${#HOSTS[@]}" ] || { echo "$N jobs but ${#HOSTS[@]} hosts"; exit 1; }

STAGE=$(mktemp -d); mkdir -p "$STAGE/code"
cp -r "$ROOT/llm" "$ROOT/synthetic_rl" "$STAGE/code/"
find "$STAGE/code" -name __pycache__ -prune -exec rm -rf {} +

i=0
while IFS= read -r CMD; do
  H=${HOSTS[$i]}; i=$((i + 1))
  cat > "$STAGE/run_$i.sh" <<EOT
#!/bin/bash
export OUT=\$HOME/rlvr_results/llm/$TAG/\$(hostname)
mkdir -p \$OUT
echo $COMMIT > \$OUT/COMMIT
[ -x /tmp/asa715/llmenv/bin/python ] || bash /tmp/asa715/code_$TAG/llm/setup_env.sh > \$OUT/setup.log 2>&1
source /tmp/asa715/llmenv/bin/activate
cp /tmp/asa715/llmenv/freeze.txt \$OUT/
export HF_HOME=/tmp/asa715/hf
export VLLM_CACHE_ROOT=/tmp/asa715/vllm-cache
cd /tmp/asa715/code_$TAG
( $CMD ) > \$OUT/job.log 2>&1 && touch \$OUT/DONE || touch \$OUT/FAILED
EOT
  ssh -n -o ConnectTimeout=20 "$H" "rm -rf /tmp/asa715/code_$TAG && mkdir -p /tmp/asa715"
  scp -q -o BatchMode=yes -r "$STAGE/code" "$H:/tmp/asa715/code_$TAG"
  scp -q -o BatchMode=yes "$STAGE/run_$i.sh" "$H:/tmp/asa715/run_$TAG.sh"
  ssh -n "$H" "chmod +x /tmp/asa715/run_$TAG.sh && tmux new -d -s llm_$TAG /tmp/asa715/run_$TAG.sh"
  echo "$H: started job $i"
done < "$JOBS"
rm -rf "$STAGE"
