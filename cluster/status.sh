#!/bin/bash
# Show load, sessions and progress of one batch on each lab machine.
# usage: cluster/status.sh <tag> host1 ... hostN
TAG=$1; shift
for H in "$@"; do
  ssh -o ConnectTimeout=20 "$H" "printf '%s load=%s users=%s results=%s failed=%s done=%s\n' \
    \$(hostname) \$(cut -d' ' -f1 /proc/loadavg) \"\$(who | cut -d' ' -f1 | sort -u | tr '\n' ',')\" \
    \$(find /tmp/asa715/rlvr/$TAG/results -name '*.json.gz' 2>/dev/null | wc -l) \
    \$(cat /tmp/asa715/rlvr/$TAG/FAILED 2>/dev/null | wc -l) \
    \$(test -f ~/rlvr_results/$TAG/\$(hostname)/DONE && echo yes || echo no)" 2>&1 | tail -1 &
done
wait
