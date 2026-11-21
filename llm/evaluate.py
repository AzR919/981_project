"""
Estimate each problem's success probability by sampling N answers with vLLM.

usage:
    python llm/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --split train --limit 2000 \\
        --n 32 --out results/llm/base_qwen05.json.gz

Writes per-problem counts {id: [correct, parsed, n]} plus the config, the
sampling settings, tokens generated and wall time. Sampling uses the
training distribution (temperature 1, top_p 1), so p-hat is the p that
GRPO sees.
"""

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.data import load_gsm8k, prompt_text, reward  # noqa: E402


def sample_counts(llm, tokenizer, problems, n, max_tokens, seed):
    from vllm import SamplingParams

    params = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=max_tokens, seed=seed)
    outputs = llm.generate([prompt_text(tokenizer, p) for p in problems], params)
    counts, tokens = {}, 0
    for problem, out in zip(problems, outputs):
        correct = parsed = 0
        for c in out.outputs:
            r, ok = reward(c.text, problem["answer"])
            correct += int(r)
            parsed += int(ok)
            tokens += len(c.token_ids)
        counts[problem["id"]] = [correct, parsed, n]
    return counts, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None, help="JSON file with a list of problem ids to evaluate")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from vllm import LLM

    problems = load_gsm8k(args.split)
    if args.ids:
        keep = set(json.load(open(args.ids)))
        problems = [p for p in problems if p["id"] in keep]
    if args.limit:
        problems = problems[: args.limit]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory, seed=args.seed,
              max_model_len=args.max_tokens + 1024)
    tokenizer = llm.get_tokenizer()
    t0 = time.time()
    counts, tokens = sample_counts(llm, tokenizer, problems, args.n, args.max_tokens, args.seed)
    seconds = time.time() - t0

    out = {"config": vars(args), "counts": counts, "tokens": tokens, "seconds": seconds,
           "tokens_per_second": tokens / seconds}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        json.dump(out, f)
    total = sum(c[0] for c in counts.values()) / sum(c[2] for c in counts.values())
    unparsed = 1 - sum(c[1] for c in counts.values()) / sum(c[2] for c in counts.values())
    print(f"{len(problems)} problems, mean p {total:.3f}, format failures {unparsed:.3f}, "
          f"{tokens} tokens in {seconds:.0f}s ({tokens / seconds:.0f} tok/s)")


if __name__ == "__main__":
    main()
