#!/bin/bash
# Feed GPU jobs to idle lab machines, one job per host at a time.
# usage: llm/queue.sh <batch> <jobs.txt> host1 ... hostN   (run from 981_project/, in the background)
# Every 2 minutes, re-reads jobs.txt (append lines to queue more work) and starts the
# next unstarted line on each idle host: no llm_* tmux session of ours and no GPU
# process of anyone's (other users run jobs without a login session). Job k runs as
# tag <batch>_<k> (results in ~/rlvr_results/llm/<batch>_<k>/<host>). The number of
# lines started is kept in <jobs.txt>.started, so a restarted queue resumes. A line
# "STOP" ends the queue once every line before it has started.
BATCH=$1; JOBS=$2; shift 2; HOSTS=("$@")
DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$JOBS.started"; [ -f "$STATE" ] || echo 0 > "$STATE"
while true; do
  for H in "${HOSTS[@]}"; do
    NEXT=$(cat "$STATE")
    LINE=$(sed -n "$((NEXT + 1))p" "$JOBS")
    [ "$LINE" = STOP ] && { echo "$(date +%H:%M) queue $BATCH: all started"; exit 0; }
    [ -z "$LINE" ] && break
    IDLE=$(ssh -n -o ConnectTimeout=20 "$H" '! tmux ls 2>/dev/null | grep -q "^llm_" && \
      [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && echo yes' 2>/dev/null)
    if [ "$IDLE" = yes ]; then
      ONE=$(mktemp); echo "$LINE" > "$ONE"
      TAG=$(printf "%s_%02d" "$BATCH" "$NEXT")
      if "$DIR/launch_gpu.sh" "$TAG" "$ONE" "$H" > /dev/null 2>&1; then
        echo $((NEXT + 1)) > "$STATE"
        echo "$(date +%H:%M) $TAG -> $H"
      fi
      rm -f "$ONE"
    fi
  done
  sleep 120
done
