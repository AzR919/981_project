#!/bin/bash
# After the six Llama calibration jobs (extra/llm/jobs/llama_c0.txt, queue tags <batch>_00 .. <batch>_05) are DONE:
# build the Llama pool and C1 sets, apply the pre-registered lr rule (PREREG_C1 amendment 1), and put the
# N = 256 base evaluations and the three C1 runs at the front of the unstarted queue.
# usage (from the queue worktree, where queue.txt lives):  extra/llm/jobs/llama_arm.sh <batch> <any-lab-host>
set -e
BATCH=$1; H=$2
L=llama; mkdir -p $L
M=meta-llama/Llama-3.2-1B-Instruct
for k in 00 01 02 03 04 05; do
  rsync -a --exclude checkpoint -e "ssh -o ConnectTimeout=20" "$H:rlvr_results/llm/${BATCH}_$k/" "$L/$k/"
  ls $L/$k/*/DONE > /dev/null || { echo "${BATCH}_$k is not DONE"; exit 1; }
done
S2=$(ls $L/00/*/base_llama_train_s2.json.gz); S3=$(ls $L/01/*/base_llama_train_s3.json.gz)
TEST=$(ls $L/02/*/base_llama_test.json.gz)
P1=$(dirname $(ls $L/03/*/after.json.gz)); P2=$(dirname $(ls $L/04/*/after.json.gz)); P3=$(dirname $(ls $L/05/*/after.json.gz))
python3 extra/llm/analyze_c0.py --name "Llama-3.2-1B-Instruct" --base "$S2" "$S3" --test "$TEST" \
  --pilot 1e-6=$P1 3e-6=$P2 1e-5=$P3 --out $L/c0_llama.md --pool extra/llm/pools/llama.json
python3 extra/llm/make_sets.py extra/llm/pools/llama.json
LR=$(python3 extra/llm/pick_lr.py --base "$S2" --ids extra/llm/pools/first1000.json 1e-6=$P1 3e-6=$P2 1e-5=$P3 2>> $L/pick_lr.log)
echo "$(date '+%m-%d %H:%M') chosen lr $LR" | tee -a $L/pick_lr.log
[ "$LR" = NONE ] && { echo "no lr meets the rule; nothing queued"; exit 0; }
EV="--ids extra/llm/pools/llama_c1_eval.json --n 256 --keep-answers"
{
  for s in 0 1; do echo "python extra/llm/evaluate.py --model $M $EV --seed $s --keep-text 4 --out \$OUT/base_llama_n256_s$s.json.gz"; done
  for s in 0 1 2; do
    echo "python extra/llm/train.py --model $M --problems extra/llm/pools/llama_c1_train.json --lr $LR --steps 500 --seed $s --micro-batch 1 --gpu-memory 0.2 --save-at 250 --check-sync --out \$OUT/run --work /tmp/asa715/w_l1 && { [ ! -d /tmp/asa715/w_l1/step_250 ] || python extra/llm/evaluate.py --model /tmp/asa715/w_l1/step_250 --ids extra/llm/pools/llama_c1_eval.json --n 32 --out \$OUT/step250.json.gz; } && python extra/llm/evaluate.py --model /tmp/asa715/w_l1/final $EV --seed 0 --out \$OUT/after_n256.json.gz && python extra/llm/evaluate.py --model /tmp/asa715/w_l1/final $EV --seed 1 --out \$OUT/after_n256_s1.json.gz && rm -rf \$OUT/run/checkpoint"
  done
} > $L/c1_llama.txt
N=$(cat queue.txt.started)
{ head -n "$N" queue.txt; cat $L/c1_llama.txt; tail -n +"$((N + 1))" queue.txt; } > queue.tmp && mv queue.tmp queue.txt
echo "queued 2 base evaluations and 3 C1 runs at lr $LR (queue lines $((N + 1))-$((N + 5)))"
