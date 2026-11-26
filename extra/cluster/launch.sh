#!/bin/bash
# Split one job list over SECB lab machines and run it in tmux (CPU only).
# usage: extra/cluster/launch.sh <tag> <jobs.jsonl> <workers-per-host> host1 ... hostN
#   Run from the laptop on SFU Wi-Fi or VPN. Host i runs jobs i, i+N, i+2N, ...
#   Results go to /tmp/asa715/rlvr/<tag>/results and are copied to the network
#   home ~/rlvr_results/<tag>/<host>/ when the host finishes, because /tmp is
#   wiped at the 03:00 reboot. Needs ~/envs/rlvr (CPU torch + numpy).
set -e
TAG=$1; JOBS=$2; WORKERS=$3; shift 3; HOSTS=("$@"); N=${#HOSTS[@]}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; D=/tmp/asa715/rlvr/$TAG   # ROOT = repo root (this script is in extra/cluster/)
LAST=$(($(wc -l < "$JOBS") - 1))
# Commit, number of modified tracked files (0 = the launch is reproducible from the commit) and job-list digest.
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD) dirty=$(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l | tr -d ' ') jobs=$(shasum -a 256 "$JOBS" | cut -c1-12)"

STAGE=$(mktemp -d); mkdir -p "$STAGE/code"
cp -r "$ROOT/synthetic_rl" "$ROOT/experiments" "$STAGE/code/"
mkdir -p "$STAGE/code/extra" && cp -r "$ROOT/extra/experiments" "$STAGE/code/extra/"   # enum mode (transformer env)
cp "$JOBS" "$STAGE/code/jobs.jsonl"
find "$STAGE/code" -name __pycache__ -prune -exec rm -rf {} +

for i in "${!HOSTS[@]}"; do
  H=${HOSTS[$i]}
  cat > "$STAGE/run_$i.sh" <<EOT
#!/bin/bash
cd $D/code && source ~/envs/rlvr/bin/activate
mkdir -p ../results ../logs
seq $i $N $LAST | xargs -P $WORKERS -I{} sh -c 'python experiments/run.py --jobs jobs.jsonl --index {} --out ../results > ../logs/{}.log 2>&1 || echo {} >> ../FAILED'
OUT=~/rlvr_results/$TAG/\$(hostname)
mkdir -p \$OUT && cp -r ../results \$OUT/
find ../logs -empty -delete; cp -r ../logs \$OUT/
[ -f ../FAILED ] && cp ../FAILED \$OUT/
echo "$COMMIT" > \$OUT/DONE
EOT
  ssh -o ConnectTimeout=20 "$H" "mkdir -p $D && rm -rf $D/code $D/results $D/logs $D/FAILED"
  scp -q -r "$STAGE/code" "$H:$D/"
  scp -q "$STAGE/run_$i.sh" "$H:$D/run.sh"
  ssh "$H" "chmod +x $D/run.sh && tmux new -d -s rlvr_$TAG $D/run.sh"
  echo "$H: jobs $i, $((i + N)), ... started ($WORKERS workers)"
done
rm -rf "$STAGE"
