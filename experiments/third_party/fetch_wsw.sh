#!/bin/bash
# Fetch the environment of Li, Pandey and Chen, "Wrong the Same Way" (NeurIPS 2026 workshop), MIT licence:
#   https://github.com/Rohan-Pandey1729/RLVR-Reward-Hacking
# Their src/ and LICENSE are copied to experiments/third_party/wsw/ (not tracked by git; shipped to the lab
# machines by cluster/launch.sh with the rest of experiments/). experiments/enum_env.py imports from there.
# usage: experiments/third_party/fetch_wsw.sh [existing clone]
set -e
PIN=05ef005c5211ad8aff5e00d417facd32f4ea0d9a
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC=${1:-$(mktemp -d)/RLVR-Reward-Hacking}
[ -d "$SRC/.git" ] || git clone -q https://github.com/Rohan-Pandey1729/RLVR-Reward-Hacking.git "$SRC"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$PIN" ] || git -C "$SRC" checkout -q "$PIN"
rm -rf "$HERE/wsw" && mkdir -p "$HERE/wsw"
cp -r "$SRC/src" "$SRC/LICENSE" "$HERE/wsw/"
find "$HERE/wsw" -name __pycache__ -prune -exec rm -rf {} +
echo "$PIN" > "$HERE/wsw/COMMIT"
echo "fetched $PIN into $HERE/wsw"
