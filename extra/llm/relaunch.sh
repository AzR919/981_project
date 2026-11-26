#!/bin/bash
# Restart the jobs of one GPU batch that neither finished nor failed (dropped
# session, reboot, the 03:00 /tmp wipe). Same arguments as launch_gpu.sh, with
# the hosts in the same order, so line i still maps to host i. A job whose tmux
# session is still alive is left alone. Training resumes from $OUT/run/checkpoint.
# usage: extra/llm/relaunch.sh <tag> <jobs.txt> host1 ... hostN [--failed]
#   --failed also restarts jobs that wrote FAILED (after fixing the cause).
set -e
TAG=$1; JOBS=$2; shift 2
FAILED=0; HOSTS=()
for a in "$@"; do [ "$a" = "--failed" ] && FAILED=1 || HOSTS+=("$a"); done
DIR="$(cd "$(dirname "$0")" && pwd)"
i=0
while IFS= read -r CMD; do
  H=${HOSTS[$i]}; i=$((i + 1))
  STATE=$(ssh -n -o ConnectTimeout=20 "$H" "O=~/rlvr_results/llm/$TAG/\$(hostname); \
    if [ -f \$O/DONE ]; then echo done; elif tmux has-session -t llm_$TAG 2>/dev/null; then echo running; \
    elif [ -f \$O/FAILED ]; then echo failed; else echo lost; fi")
  if [ "$STATE" = lost ] || { [ "$STATE" = failed ] && [ $FAILED = 1 ]; }; then
    ssh -n "$H" "rm -f ~/rlvr_results/llm/$TAG/\$(hostname)/FAILED"
    ONE=$(mktemp); echo "$CMD" > "$ONE"
    "$DIR/launch_gpu.sh" "$TAG" "$ONE" "$H"
    rm -f "$ONE"
  else
    echo "$H: $STATE"
  fi
done < "$JOBS"
