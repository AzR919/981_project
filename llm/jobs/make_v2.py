"""
Write the job lists for the second GPU round (PREREG_C1 / PREREG_C2 amendment 1).

usage (from 981_project/):  python llm/jobs/make_v2.py

One shell command per line, run by llm/queue.sh or llm/launch_gpu.sh ($OUT is the job's result folder).
  llama_c0.txt    Llama-3.2-1B calibration: base evaluations (pool seeds 2 and 3, test set) and 3 lr pilots
  c2_v2.txt       C2 base hint evaluations, then the 16 main runs with hint evaluations
  c2_extras.txt   matched-minority arm (12), samples-per-step arm (6), long arm (2), rejection-sampling control (6)
  smol_diag.txt   SmolLM2 collapse diagnostic: drift logging with and without truncated importance weights
"""

import os

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
LLAMA = "meta-llama/Llama-3.2-1B-Instruct"
SMOL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
P = "llm/pools/c2"
ALL = f"--problems-file {P}/all.json"
HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, out, n, extra=""):
    return f"python llm/evaluate.py --model {model} {extra} --n {n} --keep-answers --out $OUT/{out}.json.gz"


def hints(model, prefix=""):
    """N = 64 on the 600 held-out problems with each hint."""
    return " && ".join(evaluate(model, f"{prefix}hint_{h}", 64, f"{ALL} --ids {P}/heldout_ids.json --hint-field hint_{h}")
                       for h in ("neutral", "first_step", "wrong_step"))


def c2_run(work, train_args, eval_ids, steps=500, save_at=(250,), mid_n=32, with_hints=False):
    """Train, evaluate saved steps (skipped if /tmp was wiped), evaluate the final model, drop the checkpoint."""
    saves = " ".join(map(str, save_at))
    line = (f"python llm/train.py --model {QWEN} {ALL} {train_args} --lr 1e-6 --steps {steps} "
            f"--save-at {saves} --check-sync --out $OUT/run --work {work}")
    for s in save_at:
        mid = evaluate(f"{work}/step_{s}", f"step{s}", mid_n, f"{ALL} --ids {eval_ids}")
        line += f" && {{ [ ! -d {work}/step_{s} ] || {mid}; }}"
    line += " && " + evaluate(f"{work}/final", "after_n256", 256, f"{ALL} --ids {eval_ids} --seed 0")
    if with_hints:
        line += " && " + hints(f"{work}/final")
    return line + " && rm -rf $OUT/run/checkpoint"


def write(name, lines):
    with open(os.path.join(HERE, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{name}: {len(lines)} jobs")


def main():
    # Llama calibration. Full fine-tuning needs micro-batch 1 and a 20% vLLM share (model probes).
    lines = [f"python llm/evaluate.py --model {LLAMA} --split train --limit 3000 --n 32 --seed {s} "
             f"--out $OUT/base_llama_train_s{s}.json.gz" for s in (2, 3)]
    lines.append(f"python llm/evaluate.py --model {LLAMA} --split test --n 32 --seed 0 --out $OUT/base_llama_test.json.gz")
    for lr in ("1e-6", "3e-6", "1e-5"):
        lines.append(f"python llm/train.py --model {LLAMA} --problems llm/pools/first1000.json --lr {lr} --steps 150 "
                     f"--micro-batch 1 --gpu-memory 0.2 --check-sync --out $OUT/run --work /tmp/asa715/w_l0 && "
                     f"python llm/evaluate.py --model /tmp/asa715/w_l0/final --ids llm/pools/first1000.json --n 32 "
                     f"--out $OUT/after.json.gz && rm -rf $OUT/run/checkpoint")
    write("llama_c0.txt", lines)

    # C2 main: the same 16 mixes as PREREG_C2 (seed 2 only at m = 0.5 and 0.95).
    lines = [hints(QWEN, prefix="base_")]
    for seed in (0, 1, 2):
        for m in ("0.5", "0.8", "0.95"):
            for major in ("each", "total"):
                if seed == 2 and m == "0.8":
                    continue
                mix = f"{P}/mix_m{m}_{major}_s{seed}"
                lines.append(c2_run("/tmp/asa715/w_c2", f"--problems {mix}_train_ids.json --seed {seed}",
                                    f"{mix}_eval_ids.json", with_hints=True))
    write("c2_v2.txt", lines)

    extras = []
    balanced = f"{P}/mix_m0.5_each_s0"     # 500 + 500 training problems; the roles differ only in the strata
    for seed in (0, 1):                     # matched-minority arm: 4 minority prompts per step in every arm
        for major, minor in (("each", "total"), ("total", "each")):
            for n_major in (4, 16, 76):
                extras.append(c2_run("/tmp/asa715/w_c2m", f"--problems {balanced}_train_ids.json --seed {seed} "
                                     f"--strata procedure:{major}={n_major},{minor}=4 --denominator 80",
                                     f"{balanced}_eval_ids.json"))
    for prompts in (16, 256):               # samples-per-step arm at m = 0.95, majority "each"
        for seed in (0, 1, 2):
            mix = f"{P}/mix_m0.95_each_s{seed}"
            extras.append(c2_run("/tmp/asa715/w_c2s", f"--problems {mix}_train_ids.json --seed {seed} "
                                 f"--prompts-per-step {prompts}", f"{mix}_eval_ids.json"))
    for major in ("each", "total"):         # long arm: plateau or loss
        mix = f"{P}/mix_m0.95_{major}_s0"
        extras.append(c2_run("/tmp/asa715/w_c2l", f"--problems {mix}_train_ids.json --seed 0", f"{mix}_eval_ids.json",
                             steps=1500, save_at=(500, 1000), mid_n=256))
    for m in ("0.5", "0.95"):               # rejection-sampling control: the model's own correct answers
        for seed in (0, 1, 2):
            mix = f"{P}/mix_m{m}_each_s{seed}"
            extras.append(
                f"python llm/evaluate.py --model {QWEN} {ALL} --ids {mix}_train_ids.json --n 16 --seed {seed + 10} "
                f"--keep-correct 4 --out $OUT/targets.json.gz && "
                f"python llm/sft.py --model {QWEN} {ALL} --problems {mix}_train_ids.json --targets $OUT/targets.json.gz "
                f"--lr 1e-6 --batch 64 --epochs 8 --seed {seed} --out $OUT/run --work /tmp/asa715/w_rft && "
                + evaluate("/tmp/asa715/w_rft/final", "after_n256", 256, f"{ALL} --ids {mix}_eval_ids.json --seed 0"))
    write("c2_extras.txt", extras)

    write("smol_diag.txt", [
        f"python llm/train.py --model {SMOL} --lora 64 --gpu-memory 0.35 --problems llm/pools/smol_c1_train.json "
        f"--lr 1e-4 --steps 200 --check-sync {tis} --out $OUT/run --work /tmp/asa715/w_sd && rm -rf $OUT/run/checkpoint"
        for tis in ("", "--tis 2.0")])


if __name__ == "__main__":
    main()
