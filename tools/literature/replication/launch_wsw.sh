#!/bin/bash
# Run the replication jobs of "Wrong the Same Way" on SECB lab CPUs, in tmux.
# usage: tools/literature/replication/launch_wsw.sh <tag> <jobs.txt> <clone at the pinned commit> <workers> host1 ... hostN
#   Host i runs jobs i, i+N, ... Each job runs in its own copy of their src/ with empty results/ and cache/,
#   under ~/envs/rlvr (torch 2.15.0, the version they pin). Results are copied to the network home
#   ~/rlvr_results/<tag>/<host>/j<index>/ as each job ends; DONE is written when the host has finished.
#   Hosts where another user is logged in are skipped at launch.
set -e
TAG=$1; JOBS=$2; CLONE=$3; WORKERS=$4; shift 4; HOSTS=("$@")
PIN=05ef005c5211ad8aff5e00d417facd32f4ea0d9a
[ "$(git -C "$CLONE" rev-parse HEAD)" = "$PIN" ] || { echo "clone is not at $PIN"; exit 1; }
[ -z "$(git -C "$CLONE" status --porcelain --untracked-files=no)" ] || { echo "clone has local changes"; exit 1; }
D=/tmp/asa715/$TAG
OK=()
for H in "${HOSTS[@]}"; do
  others=$(ssh -n -o ConnectTimeout=15 "$H" 'who | awk "{print \$1}" | sort -u | grep -vx "$(whoami)" | tr "\n" " "' 2>/dev/null) || { echo "$H: unreachable, skipped"; continue; }
  [ -n "$others" ] && { echo "$H: skipped (logged in: $others)"; continue; }
  OK+=("$H")
done
N=${#OK[@]}; LAST=$(($(wc -l < "$JOBS") - 1))
[ "$N" -gt 0 ] || { echo "no host available"; exit 1; }
STAGE=$(mktemp -d); mkdir -p "$STAGE/code"
cp -r "$CLONE/src" "$CLONE/LICENSE" "$STAGE/code/"; cp "$JOBS" "$STAGE/code/jobs.txt"; echo "$PIN" > "$STAGE/code/COMMIT"
find "$STAGE/code" -name __pycache__ -prune -exec rm -rf {} +
cat > "$STAGE/code/job.sh" <<'EOT'
#!/bin/bash
# job.sh <index>: run one job in a fresh copy of the code.
i=$1; D=$(cd "$(dirname "$0")/.." && pwd); W=$D/run/j$i
rm -rf "$W" && mkdir -p "$W/results" "$W/cache" && cp -r "$D/code/src" "$W/src"
CMD=$(sed -n "$((i + 1))p" "$D/code/jobs.txt")
source ~/envs/rlvr/bin/activate
cd "$W/src" && ( echo "$CMD"; date; eval "$CMD"; echo "exit $?"; date ) > "$W/job.log" 2>&1
OUT=~/rlvr_results/$(basename "$D")/$(hostname)/j$i
mkdir -p "$OUT" && cp -r "$W/results" "$W/job.log" "$OUT/"
grep -q "^exit 0$" "$W/job.log" || echo "$i" >> "$D/FAILED"
rm -rf "$W"
EOT
chmod +x "$STAGE/code/job.sh"
for i in "${!OK[@]}"; do
  H=${OK[$i]}
  cat > "$STAGE/run_$i.sh" <<EOT
#!/bin/bash
cd $D && seq $i $N $LAST | xargs -P $WORKERS -I{} code/job.sh {}
OUT=~/rlvr_results/$TAG/\$(hostname); mkdir -p \$OUT
[ -f FAILED ] && cp FAILED \$OUT/
cp code/COMMIT \$OUT/DONE
EOT
  ssh -o ConnectTimeout=20 "$H" "mkdir -p $D && rm -rf $D/code $D/run $D/FAILED"
  scp -q -r "$STAGE/code" "$H:$D/"
  scp -q "$STAGE/run_$i.sh" "$H:$D/run.sh"
  ssh "$H" "chmod +x $D/run.sh && tmux new -d -s rep_$TAG $D/run.sh"
  echo "$H: jobs $i, $((i + N)), ... started ($WORKERS workers)"
done
rm -rf "$STAGE"
