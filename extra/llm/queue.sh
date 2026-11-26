#!/bin/bash
# Feed GPU jobs to idle lab machines, one job per host at a time.
# usage: extra/llm/queue.sh <batch> <jobs.txt> host1 ... hostN   (run from 981_project/, in the background)
# Every 2 minutes, re-reads jobs.txt (append lines to queue more work) and starts the
# next unstarted line on each idle host: no llm_* tmux session of ours and no GPU
# process of anyone's (other users run jobs without a login session). Job k runs as
# tag <batch>_<k> (results in ~/rlvr_results/llm/<batch>_<k>/<host>). The number of
# lines started is kept in <jobs.txt>.started, so a restarted queue resumes. A line
# "STOP" ends the queue once every line before it has started.
#
# Every launch is recorded in <jobs.txt>.placed ("tag host"). An idle host whose last job
# left neither DONE nor FAILED was cut off (a reboot empties /tmp and kills the session):
# that job is started again on the same host, at most twice, before any new line; training
# resumes from the checkpoint on the network home. To stop a job by hand without having it
# restarted, touch FAILED in its result folder. Lines already started must not be moved.
#
# Sharing the lab: a host counts as idle only if no other user is logged in on it, and at most
# MAX_JOBS hosts (environment variable, default: all) run one of our jobs at a time.
BATCH=$1; JOBS=$2; shift 2; HOSTS=("$@")
DIR="$(cd "$(dirname "$0")" && pwd)"
STATE="$JOBS.started"; [ -f "$STATE" ] || echo 0 > "$STATE"
PLACED="$JOBS.placed"; touch "$PLACED"
MAX_JOBS=${MAX_JOBS:-${#HOSTS[@]}}
echo "$(date +%H:%M) queue $BATCH: ${#HOSTS[@]} hosts, at most $MAX_JOBS running at a time, skipping hosts with another user logged in"

start() {  # start <tag> <command line> <host>
  local ONE RC; ONE=$(mktemp); echo "$2" > "$ONE"
  "$DIR/launch_gpu.sh" "$1" "$ONE" "$3" > /dev/null 2>&1; RC=$?
  rm -f "$ONE"; return $RC
}

while true; do
  # Hosts running one of our jobs now (a host that does not answer counts as busy, to stay under the cap).
  BUSY=0
  for H in "${HOSTS[@]}"; do
    R=$(ssh -n -o ConnectTimeout=20 "$H" 'tmux ls 2>/dev/null | grep -q "^llm_" && echo yes || echo no' 2>/dev/null)
    [ "$R" = no ] || BUSY=$((BUSY + 1))
  done
  for H in "${HOSTS[@]}"; do
    [ "$BUSY" -lt "$MAX_JOBS" ] || break
    IDLE=$(ssh -n -o ConnectTimeout=20 "$H" '! tmux ls 2>/dev/null | grep -q "^llm_" && \
      ! who | awk "{print \$1}" | grep -qvx "$(whoami)" && \
      [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && echo yes' 2>/dev/null)
    [ "$IDLE" = yes ] || continue
    LAST=$(awk -v h="$H" '$2 == h {t = $1} END {print t}' "$PLACED")
    if [ -n "$LAST" ] && [ "$(grep -c "^$LAST $H\$" "$PLACED")" -le 2 ]; then
      ENDED=$(ssh -n -o ConnectTimeout=20 "$H" "ls ~/rlvr_results/llm/$LAST/\$(hostname)/ 2>/dev/null | grep -cE '^(DONE|FAILED)\$'" 2>/dev/null)
      [ -z "$ENDED" ] && continue                      # host did not answer: decide next round
      if [ "$ENDED" = 0 ]; then
        K=$((10#${LAST##*_}))
        if start "$LAST" "$(sed -n "$((K + 1))p" "$JOBS")" "$H"; then
          BUSY=$((BUSY + 1))
          echo "$LAST $H" >> "$PLACED"
          echo "$(date +%H:%M) $LAST -> $H (restarted: it had not finished)"
        fi
        continue
      fi
    fi
    NEXT=$(cat "$STATE")
    LINE=$(sed -n "$((NEXT + 1))p" "$JOBS")
    [ "$LINE" = STOP ] && { echo "$(date +%H:%M) queue $BATCH: all started"; exit 0; }
    [ -z "$LINE" ] && continue
    TAG=$(printf "%s_%02d" "$BATCH" "$NEXT")
    if start "$TAG" "$LINE" "$H"; then
      BUSY=$((BUSY + 1))
      echo $((NEXT + 1)) > "$STATE"
      echo "$TAG $H" >> "$PLACED"
      echo "$(date +%H:%M) $TAG -> $H"
    fi
  done
  sleep 120
done
