"""
GRPO-family training of a small LLM on binary-reward math problems.

usage:
    python llm/train.py --model Qwen/Qwen2.5-0.5B-Instruct --problems ids.json --rule grpo \\
        --steps 300 --lr 1e-6 --out ~/rlvr_results/llm/<run> --work /tmp/asa715/runs/<run>

The update matches the toy (synthetic_rl/trainer.py): advantages come from
calculate_advantages, and the loss is -(A * log pi(completion)).mean() over
all B*G samples, where log pi sums token log-probabilities (no length
normalization). One update per batch, so the samples are on-policy and no
importance ratio is needed. Optional: reverse KL to the initial model (k3
estimator per token, summed over the completion), reward flips, LoRA.

Sampling uses vLLM in the same process; after every update the trained
weights (fp32 master copy, cast to bf16) are loaded into vLLM.

Checkpoints (model, optimizer, RNG, step) go to --work every
--save-minutes and are mirrored to --out, so a run resumes after the 03:00
reboot. The final model is saved under --work/final for evaluation.
"""

import argparse
import gzip
import json
import os
import random
import shutil
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # keep vLLM in-process for weight sync
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.data import load_gsm8k, prompt_text, reward  # noqa: E402
from synthetic_rl.trainer import calculate_advantages  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--problems", required=True, help="JSON list of GSM8K problem ids to train on")
    p.add_argument("--split", default="train")
    p.add_argument("--rule", default="grpo")
    p.add_argument("--passk", type=int, default=None)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=64)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--reward-flip", type=float, default=0.0)
    p.add_argument("--lora", type=int, default=0, help="LoRA rank; 0 = full fine-tuning")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--gpu-memory", type=float, default=0.2)
    p.add_argument("--save-minutes", type=float, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-at", type=int, nargs="*", default=[],
                   help="steps at which to export a bf16 model to <work>/step_<n> (for mid-run evaluation)")
    p.add_argument("--check-sync", action="store_true",
                   help="log vLLM and policy log-probs of one sampled answer after each of the first 3 syncs")
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    return p.parse_args()


# ---------------------------------------------------------------- model

def build_policy(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.lora:
        from peft import LoraConfig, get_peft_model

        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
        config = LoraConfig(r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.0,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(base, config)
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()  # fp32 adapters
    else:
        # fp32 master weights: at lr ~1e-6 a bf16 update would round to zero.
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer


def reference_model(args, model):
    """Frozen initial policy for the KL term (None when beta = 0)."""
    if args.beta <= 0:
        return None
    if args.lora:
        return "adapter-disabled"
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def vllm_weights(model, lora):
    """(name, bf16 tensor) pairs in the base model's naming, LoRA merged in.

    The merge is computed out of place in fp32 and every tensor is a fresh copy.
    (An earlier version merged in place and returned bf16 base tensors uncopied,
    so unmerge_adapter() reverted them before vLLM loaded them and vLLM kept
    sampling from the base model.)
    """
    if lora:
        deltas = {name + ".base_layer.weight": module.get_delta_weight("default")
                  for name, module in model.named_modules()
                  if hasattr(module, "get_delta_weight") and "default" in getattr(module, "lora_A", {})}
        state = []
        for name, tensor in model.state_dict().items():
            if "lora_" in name:
                continue
            if name in deltas:
                tensor = tensor.float() + deltas[name].float()
            state.append((name.replace("base_model.model.", "").replace(".base_layer", ""),
                          tensor.to(torch.bfloat16, copy=True)))
        return state
    return [(n, t.to(torch.bfloat16, copy=True)) for n, t in model.state_dict().items()]


def sync_check(llm, model, tokenizer, problem, pad, seed):
    """Sum of completion log-probs of one sampled answer under vLLM and under the policy."""
    from vllm import SamplingParams

    params = SamplingParams(n=1, temperature=1.0, max_tokens=64, logprobs=0, seed=seed)
    out = llm.generate([prompt_text(tokenizer, problem)], params, use_tqdm=False)[0]
    c = out.outputs[0]
    vllm_sum = float(sum(lp[t].logprob for lp, t in zip(c.logprobs, c.token_ids)))
    with torch.no_grad():
        hf_sum, _ = completion_logprobs(model, [list(out.prompt_token_ids) + list(c.token_ids)],
                                        [len(out.prompt_token_ids)], pad)
    return {"vllm_logp": vllm_sum, "policy_logp": float(hf_sum[0]), "tokens": len(c.token_ids)}


def sync_to_vllm(llm, weights):
    """Load weights into the in-process vLLM model (API differs across vLLM versions)."""
    if hasattr(llm, "apply_model"):
        llm.apply_model(lambda m: m.load_weights(weights))
        return
    engine = llm.llm_engine
    worker = engine.model_executor.driver_worker
    runner = getattr(worker, "worker", worker).model_runner
    runner.model.load_weights(weights)


# ---------------------------------------------------------------- loss

def token_logprobs(logits, targets):
    """log softmax(logits)[target] in fp32, without materialising a second vocabulary-sized tensor."""
    logits = logits.float()
    return logits.gather(-1, targets[..., None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def completion_logprobs(model, sequences, prompt_lens, pad_id, ref=None, adapter_off=False):
    """Per-sequence sum of completion token log-probs (and per-token KL k3 vs ref)."""
    lengths = [len(s) for s in sequences]
    width = max(lengths)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device="cuda")
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, s in enumerate(sequences):
        ids[i, : len(s)] = torch.tensor(s, device="cuda")
        mask[i, prompt_lens[i]: len(s)] = True
    attention = torch.arange(width, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
    targets = ids[:, 1:]
    token_mask = mask[:, 1:]
    logp = token_logprobs(logits, targets)
    seq_logp = (logp * token_mask).sum(dim=1)
    kl = None
    if ref is not None:
        with torch.no_grad():
            if adapter_off:
                with model.disable_adapter(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ref_logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ref_logits = ref(input_ids=ids, attention_mask=attention).logits[:, :-1]
            ref_logp = token_logprobs(ref_logits, targets)
        delta = ref_logp - logp
        kl = ((torch.exp(delta) - delta - 1) * token_mask).sum(dim=1)  # k3, summed over the completion
    return seq_logp, kl


# ---------------------------------------------------------------- checkpoints

def save_checkpoint(path, model, optimizer, step, rng_state, lora):
    tmp = path + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad} if lora \
        else {n: p.detach().cpu() for n, p in model.state_dict().items()}
    torch.save({"weights": trainable, "optimizer": optimizer.state_dict(), "step": step, "rng": rng_state},
               os.path.join(tmp, "state.pt"))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, lora):
    state = torch.load(os.path.join(path, "state.pt"), map_location="cpu", weights_only=False)
    if lora:
        params = dict(model.named_parameters())
        for n, t in state["weights"].items():
            params[n].data.copy_(t.to(params[n].device))
    else:
        model.load_state_dict(state["weights"])
    optimizer.load_state_dict(state["optimizer"])
    return state["step"], state["rng"]


def export(model, tokenizer, path, lora):
    """Save a bf16 copy of the current policy (LoRA merged) that vLLM can load; training state is untouched."""
    inner = model.base_model.model if lora else model
    inner.save_pretrained(path, state_dict=dict(vllm_weights(model, lora)))
    tokenizer.save_pretrained(path)


# ---------------------------------------------------------------- main

def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from vllm import LLM, SamplingParams

    keep = set(json.load(open(args.problems)))
    problems = [p for p in load_gsm8k(args.split) if p["id"] in keep]
    model, tokenizer = build_policy(args)
    ref = reference_model(args, model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.lora:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    else:
        # 8-bit Adam state (1 GB instead of 4 GB for 0.5B) so training fits beside vLLM on 24 GB.
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    # Prompts are ~200 tokens and completions <= max_tokens; a short context keeps vLLM's KV reserve small.
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory, seed=args.seed,
              enable_prefix_caching=True, max_model_len=args.max_tokens + 1024)

    rng = np.random.default_rng(args.seed)
    step = 0
    ckpt = os.path.join(args.out, "checkpoint")
    if os.path.exists(os.path.join(ckpt, "state.pt")):
        step, rng_state = load_checkpoint(ckpt, model, optimizer, args.lora)
        rng.bit_generator.state = rng_state
        print(f"resumed at step {step}", flush=True)
    sync_to_vllm(llm, vllm_weights(model, args.lora))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.check_sync:
        print(json.dumps({"step": step, "sync_check": sync_check(llm, model, tokenizer, problems[0], pad, 0)}),
              flush=True)

    log_path = os.path.join(args.out, "log.jsonl")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    last_save = time.time()

    while step < args.steps:
        t0 = time.time()
        batch = [problems[i] for i in rng.choice(len(problems), size=args.prompts_per_step, replace=False)]
        prompts = [prompt_text(tokenizer, p) for p in batch]
        params = SamplingParams(n=args.G, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                                seed=int(rng.integers(2**31)))
        outputs = llm.generate(prompts, params, use_tqdm=False)
        t_sample = time.time() - t0

        rewards = np.zeros((len(batch), args.G))
        parsed = np.zeros_like(rewards)
        sequences, prompt_lens, lengths = [], [], []
        for i, (problem, out) in enumerate(zip(batch, outputs)):
            for g, c in enumerate(out.outputs):
                rewards[i, g], parsed[i, g] = reward(c.text, problem["answer"])
                sequences.append(list(out.prompt_token_ids) + list(c.token_ids))
                prompt_lens.append(len(out.prompt_token_ids))
                lengths.append(len(c.token_ids))
        clean = rewards.copy()
        if args.reward_flip > 0:
            flips = rng.random(rewards.shape) < args.reward_flip
            rewards = np.where(flips, 1 - rewards, rewards)
        A = calculate_advantages(torch.tensor(rewards), kind=args.rule, passk=args.passk).flatten().float().cuda()

        # Samples with zero advantage contribute nothing to the policy term; skip them unless KL needs them.
        active = torch.nonzero((A != 0) | torch.tensor(args.beta > 0, device="cuda")).flatten().tolist()
        n_total = len(sequences)
        optimizer.zero_grad(set_to_none=True)
        kl_sum = 0.0
        for start in range(0, len(active), args.micro_batch):
            idx = active[start: start + args.micro_batch]
            seq_logp, kl = completion_logprobs(model, [sequences[i] for i in idx], [prompt_lens[i] for i in idx],
                                               pad, ref=None if args.beta <= 0 else (model if args.lora else ref),
                                               adapter_off=bool(args.lora))
            loss = -(A[idx] * seq_logp).sum() / n_total
            if kl is not None:
                loss = loss + args.beta * kl.sum() / n_total
                kl_sum += kl.sum().item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1e9)  # norm only, no clipping
        grad_norm = float(sum(p.grad.float().norm() ** 2 for p in model.parameters()
                              if p.requires_grad and p.grad is not None) ** 0.5)
        optimizer.step()
        sync_to_vllm(llm, vllm_weights(model, args.lora))
        step += 1

        group = clean.mean(axis=1)
        record = {"step": step, "reward": float(clean.mean()), "reward_used": float(rewards.mean()),
                  "informative": float(((group > 0) & (group < 1)).mean()), "parsed": float(parsed.mean()),
                  "length": float(np.mean(lengths)), "active": len(active), "grad_norm": grad_norm,
                  "kl": kl_sum / n_total, "t_sample": t_sample, "t_step": time.time() - t0,
                  "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        if args.check_sync and step <= 3:
            record["sync_check"] = sync_check(llm, model, tokenizer, problems[0], pad, 0)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if step in args.save_at:
            export(model, tokenizer, os.path.join(args.work, f"step_{step}"), args.lora)

        if time.time() - last_save > 60 * args.save_minutes or step == args.steps:
            save_checkpoint(ckpt, model, optimizer, step, rng.bit_generator.state, args.lora)
            last_save = time.time()

    final = os.path.join(args.work, "final")
    export(model, tokenizer, final, args.lora)
    # The resume checkpoint (~6 GB for 0.5B with fp32 Adam) lives on the shared home disk.
    shutil.rmtree(ckpt, ignore_errors=True)
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
