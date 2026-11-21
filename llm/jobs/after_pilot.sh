#!/bin/bash
# Wait for the four SmolLM2 pilots, pick the lr by the pre-registered rule, then queue
# C1 SmolLM2 (3 seeds) and after it C7. Run from the queue worktree (~/rlvr-queue).
R=$HOME/rlvr-queue/pilots; mkdir -p $R
while true; do
  rsync -a secb1010-a01:rlvr_results/llm/c0d/ secb1010-a01:rlvr_results/llm/c0e/ $R/ 2>/dev/null
  n=$(ls $R/secb1010u-a12/after.json.gz $R/secb1010u-b12/after.json.gz $R/secb1010u-c01/after.json.gz $R/secb1010u-a01/after.json.gz 2>/dev/null | wc -l)
  [ "$n" -eq 4 ] && break
  sleep 300
done
LR=$(python3 llm/pick_lr.py --base ../Desktop/CMPT\ 981/981_project/results/llm/c0/secb1010u-a12/base_smol_train_s0.json.gz --ids llm/pools/smol_c1_train.json 1e-5=$R/secb1010u-a12 3e-5=$R/secb1010u-b12 1e-4=$R/secb1010u-c01 3e-4=$R/secb1010u-a01 2>> after_pilot.log)
echo "$(date +%H:%M) chosen lr $LR" >> after_pilot.log
if [ "$LR" != NONE ]; then
  for s in 0 1 2; do
    echo "python llm/train.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --lora 64 --gpu-memory 0.35 --problems llm/pools/smol_c1_train.json --lr $LR --steps 500 --seed $s --save-at 250 --check-sync --out \$OUT/run --work /tmp/asa715/w_s && { [ ! -d /tmp/asa715/w_s/step_250 ] || python llm/evaluate.py --model /tmp/asa715/w_s/step_250 --ids llm/pools/smol_c1_eval.json --n 32 --out \$OUT/step250.json.gz; } && python llm/evaluate.py --model /tmp/asa715/w_s/final --ids llm/pools/smol_c1_eval.json --n 256 --seed 0 --keep-answers --out \$OUT/after_n256.json.gz && rm -rf \$OUT/run/checkpoint" >> queue.txt
  done
fi
cat llm/jobs/c7.txt >> queue.txt
echo "$(date +%H:%M) queued C1 SmolLM2 (lr $LR) and C7" >> after_pilot.log
