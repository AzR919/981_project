"""
Estimate each problem's success probability by sampling N answers with vLLM.

usage:
    python llm/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --split train --limit 2000 \\
        --n 32 --out results/llm/base_qwen05.json.gz

Writes per-problem counts {id: [correct, parsed, n]} plus the config, the
sampling settings, tokens generated and wall time. --keep-answers adds a
histogram of parsed answers per problem (C2: did a lost problem switch to the
other procedure's answer?); --keep-text K keeps K raw completions per problem
(C3 strategy clusters). Sampling uses the
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

from llm.data import extract_answer, load_gsm8k, prompt_text, reward  # noqa: E402


def sample_counts(llm, tokenizer, problems, n, max_tokens, seed, keep_answers=False, keep_text=0, lora_request=None,
                  keep_correct=0):
    from vllm import SamplingParams

    params = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=max_tokens, seed=seed)
    outputs = llm.generate([prompt_text(tokenizer, p) for p in problems], params, lora_request=lora_request)
    counts, answers, texts, tokens, correct_texts = {}, {}, {}, 0, {}
    for problem, out in zip(problems, outputs):
        correct = parsed = 0
        hist = {}
        kept = []
        for c in out.outputs:
            r, ok = reward(c.text, problem["answer"])
            if r and len(kept) < keep_correct:
                kept.append(c.text)
            correct += int(r)
            parsed += int(ok)
            tokens += len(c.token_ids)
            if keep_answers:
                key = extract_answer(c.text) or "unparsed"
                hist[key] = hist.get(key, 0) + 1
        counts[problem["id"]] = [correct, parsed, n]
        if keep_answers:
            answers[problem["id"]] = hist
        if keep_text:
            texts[problem["id"]] = [c.text for c in out.outputs[:keep_text]]
        if keep_correct and kept:
            correct_texts[problem["id"]] = kept
    return counts, answers, texts, tokens, correct_texts


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
    parser.add_argument("--problems-file", default=None,
                        help="JSON list of problem dicts {id, question, answer} instead of GSM8K (C2)")
    parser.add_argument("--keep-answers", action="store_true")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (train.py --sync adapter) on --model")
    parser.add_argument("--quant", choices=("none", "int8"), default="none", help="vLLM per-channel int8 weights")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--keep-text", type=int, default=0)
    parser.add_argument("--keep-correct", type=int, default=0,
                        help="store up to K correct completions per problem (targets for the rejection-sampling control)")
    parser.add_argument("--hint-field", default=None,
                        help="append this field of each problem to its question as a hint (C2 access test)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from vllm import LLM

    problems = json.load(open(args.problems_file)) if args.problems_file else load_gsm8k(args.split)
    if args.ids:
        keep = set(json.load(open(args.ids)))
        problems = [p for p in problems if p["id"] in keep]
    if args.limit:
        problems = problems[: args.limit]
    if args.hint_field:
        problems = [dict(q, question=f"{q['question']}\n\nHint: {q[args.hint_field]}") for q in problems]

    kw = dict(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory, seed=args.seed,
              max_model_len=args.max_tokens + 1024)
    if args.quant == "int8":
        kw["quantization"] = "int8_per_channel_weight_only"
    request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest

        kw.update(enable_lora=True, max_lora_rank=args.lora_rank, max_loras=1)
        request = LoRARequest("eval", 1, args.adapter)
    llm = LLM(**kw)
    tokenizer = llm.get_tokenizer()
    t0 = time.time()
    counts, answers, texts, tokens, correct_texts = sample_counts(
        llm, tokenizer, problems, args.n, args.max_tokens, args.seed, args.keep_answers, args.keep_text, request,
        args.keep_correct)
    seconds = time.time() - t0

    out = {"config": vars(args), "counts": counts, "tokens": tokens, "seconds": seconds,
           "tokens_per_second": tokens / seconds}
    if args.keep_answers:
        out["answers"] = answers
    if args.keep_text:
        out["texts"] = texts
    if args.keep_correct:
        out["correct_texts"] = correct_texts
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        json.dump(out, f)
    total = sum(c[0] for c in counts.values()) / sum(c[2] for c in counts.values())
    unparsed = 1 - sum(c[1] for c in counts.values()) / sum(c[2] for c in counts.values())
    print(f"{len(problems)} problems, mean p {total:.3f}, format failures {unparsed:.3f}, "
          f"{tokens} tokens in {seconds:.0f}s ({tokens / seconds:.0f} tok/s)")


if __name__ == "__main__":
    main()
