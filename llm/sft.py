"""
C2 control: supervised fine-tuning on the same imbalanced mix that GRPO trains on.

usage:
    python llm/sft.py --model Qwen/Qwen2.5-0.5B-Instruct --problems-file mix.json \\
        --epochs 2 --lr 1e-5 --batch 16 --out $OUT/run --work /tmp/asa715/w_sft

Cross-entropy on each problem's worked solution (gen_procedures.worked), prompt
tokens masked. Same policy setup and bf16 export as train.py, so the result is
evaluated exactly like a GRPO run.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.data import prompt_text  # noqa: E402
from llm.train import build_policy, export, token_logprobs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--problems-file", required=True)
    p.add_argument("--problems", default=None, help="JSON list of ids to train on (default: all)")
    p.add_argument("--targets", default=None,
                   help="evaluate.py output with correct_texts: train on the model's own correct completions "
                        "(rejection sampling) instead of the templated solutions")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--lora", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    problems = json.load(open(args.problems_file))
    if args.problems:
        keep = set(json.load(open(args.problems)))
        problems = [q for q in problems if q["id"] in keep]
    model, tokenizer = build_policy(args)
    trainable = [q for q in model.parameters() if q.requires_grad]
    if args.lora:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    else:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=0.0)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    targets = None
    if args.targets:
        import gzip

        targets = json.load(gzip.open(args.targets, "rt"))["correct_texts"]
    examples = []
    for q in problems:
        prompt = tokenizer(prompt_text(tokenizer, q), add_special_tokens=False).input_ids
        # Problems the base model never solved contribute nothing under rejection sampling.
        for text in ([q["solution"]] if targets is None else targets.get(q["id"], [])):
            answer = tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            examples.append((prompt + answer, len(prompt)))
    print(f"{len(examples)} training examples from {len(problems)} problems", flush=True)

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(examples))
        for start in range(0, len(order), args.batch):
            t0 = time.time()
            batch = [examples[i] for i in order[start: start + args.batch]]
            n_tokens = sum(len(s) - k for s, k in batch)
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for m in range(0, len(batch), args.micro_batch):
                part = batch[m: m + args.micro_batch]
                width = max(len(s) for s, _ in part)
                ids = torch.full((len(part), width), pad, dtype=torch.long, device="cuda")
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for j, (s, k) in enumerate(part):
                    ids[j, : len(s)] = torch.tensor(s, device="cuda")
                    mask[j, k: len(s)] = True
                attention = ids != pad
                for j, (s, _) in enumerate(part):
                    attention[j, : len(s)] = True
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
                logp = token_logprobs(logits, ids[:, 1:])
                loss = -(logp * mask[:, 1:]).sum() / n_tokens
                loss.backward()
                total += loss.item()
            optimizer.step()
            step += 1
            log.write(json.dumps({"step": step, "epoch": epoch, "loss": total, "t_step": time.time() - t0}) + "\n")
            log.flush()
    export(model, tokenizer, os.path.join(args.work, "final"), args.lora)
    print(f"saved {os.path.join(args.work, 'final')} after {step} steps", flush=True)


if __name__ == "__main__":
    main()
