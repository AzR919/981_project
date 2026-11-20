#!/bin/bash
# Copy one batch's results from the network home to the laptop.
# usage: cluster/fetch.sh <tag> <local-folder> <any-lab-host>
# The home is shared by all lab machines, so one host serves every result.
TAG=$1; OUT=$2; H=$3
mkdir -p "$OUT"
rsync -a -e "ssh -o ConnectTimeout=20" "$H:rlvr_results/$TAG/" "$OUT/"
echo "$TAG: hosts done $(ls "$OUT"/*/DONE 2>/dev/null | wc -l), results $(find "$OUT" -name '*.json.gz' | wc -l), failed jobs $(cat "$OUT"/*/FAILED 2>/dev/null | wc -l)"
