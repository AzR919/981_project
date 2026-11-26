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
    p.add_argument("--problems", default=None, help="JSON list of problem ids to train on (default: all)")
    p.add_argument("--problems-file", default=None,
                   help="JSON list of problem dicts {id, question, answer} instead of GSM8K (C2)")
    p.add_argument("--split", default="train")
    p.add_argument("--rule", default="grpo")
    p.add_argument("--passk", type=int, default=None)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=64)
    p.add_argument("--strata", default=None,
                   help="fixed batch composition by a problem field, e.g. procedure:each=76,total=4 (overrides "
                        "--prompts-per-step); C2 matched-minority arm")
    p.add_argument("--denominator", type=int, default=None,
                   help="number of prompts the loss is divided by (default: prompts in the batch)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--kl", choices=("reverse", "forward"), default="reverse",
                   help="KL direction of the penalty gradient: reverse KL(pi||ref) or forward KL(ref||pi)")
    p.add_argument("--drift-log", action=argparse.BooleanOptionalAction, default=True,
                   help="log the sampler-trainer mismatch each step (vLLM log-probs of the sampled answers)")
    p.add_argument("--tis", type=float, default=None,
                   help="truncated importance weights min(pi_trainer / pi_sampler, C) on the policy term (off by default)")
    p.add_argument("--agg", choices=("sequence", "token"), default="sequence",
                   help="sequence: sum_i A_i log pi(y_i) / n; token: / total completion tokens (DAPO), rescaled "
                        "by the step-1 mean length so only the length weighting changes")
    p.add_argument("--reward-flip", type=float, default=0.0)
    p.add_argument("--lora", type=int, default=0, help="LoRA rank; 0 = full fine-tuning")
    p.add_argument("--quant", choices=("none", "int8"), default="none",
                   help="int8: per-channel int8 base for training (bitsandbytes 8-bit) and sampling (vLLM "
                        "int8_per_channel_weight_only); needs --lora. vLLM 0.34 has no bitsandbytes 4-bit.")
    p.add_argument("--sync", choices=("merged", "adapter"), default="merged",
                   help="merged: load merged bf16 weights into vLLM; adapter: vLLM LoRA, adapter saved each step")
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
    args = p.parse_args()
    if args.quant != "none" and not (args.lora and args.sync == "adapter"):
        p.error("--quant needs --lora and --sync adapter (a quantized vLLM model cannot load merged weights)")
    if args.sync == "adapter" and not args.lora:
        p.error("--sync adapter needs --lora")
    return args


# ---------------------------------------------------------------- model

def build_policy(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.lora:
        from peft import LoraConfig, get_peft_model

        if args.quant == "int8":
            from peft import prepare_model_for_kbit_training
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(load_in_8bit=True)
            base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                        quantization_config=quant, device_map={"": 0})
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        else:
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


class Sampler:
    """vLLM in this process, kept in step with the policy.

    merged:  the merged bf16 weights are loaded into vLLM after every update.
    adapter: vLLM serves the base model with LoRA enabled; after every update the
             adapter is saved and the next generate() uses it (needed with a
             bitsandbytes-quantized vLLM model, which cannot load merged weights).
    """

    def __init__(self, args):
        from vllm import LLM

        kw = dict(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory, seed=args.seed,
                  enable_prefix_caching=True, max_model_len=args.max_tokens + 1024)
        if args.quant == "int8":
            kw["quantization"] = "int8_per_channel_weight_only"
        if args.sync == "adapter":
            kw.update(enable_lora=True, max_lora_rank=args.lora, max_loras=1)
        # Prompts are ~200 tokens and completions <= max_tokens; a short context keeps vLLM's KV reserve small.
        self.llm = LLM(**kw)
        self.args, self.version, self.request = args, 0, None
        self.adapters = os.path.join(args.work, "adapters")

    def sync(self, model):
        if self.args.sync == "merged":
            sync_to_vllm(self.llm, vllm_weights(model, self.args.lora))
            return
        from vllm.lora.request import LoRARequest

        self.version += 1
        path = os.path.join(self.adapters, f"v{self.version}")
        model.save_pretrained(path)
        self.request = LoRARequest(f"v{self.version}", self.version, path)
        shutil.rmtree(os.path.join(self.adapters, f"v{self.version - 2}"), ignore_errors=True)

    def generate(self, prompts, params):
        return self.llm.generate(prompts, params, use_tqdm=False, lora_request=self.request)


def sync_check(sampler, model, tokenizer, problem, pad, seed):
    """Sum of completion log-probs of one sampled answer under vLLM and under the policy."""
    from vllm import SamplingParams

    params = SamplingParams(n=1, temperature=1.0, max_tokens=64, logprobs=0, seed=seed)
    out = sampler.generate([prompt_text(tokenizer, problem)], params)[0]
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


KL_LOG_RATIO_BOUND = 20.0


def kl_terms(logp, ref_logp, direction):
    """Per-token KL loss term and value. Samples come from pi, so only log pi carries gradient:
    differentiating k2 = (log pi - log ref)^2 / 2 gives the reverse-KL gradient, differentiating
    k3 = rho - log rho - 1 (rho = ref / pi) gives the forward-KL gradient (tested in
    tests/test_llm_kl.py). The value logged is k3, an unbiased estimate of the reverse KL.

    The log-ratio is bounded to [-20, 20]. Without the bound, one sampled token that the trainer
    scores as (nearly) impossible makes exp(delta) overflow or delta infinite, and the whole update
    becomes NaN (this destroyed two C7/C8 runs on 2026-11-23). Tokens beyond the bound contribute no
    KL gradient."""
    delta = (ref_logp - logp).clamp(-KL_LOG_RATIO_BOUND, KL_LOG_RATIO_BOUND)
    value = (torch.exp(delta) - delta - 1).detach()
    loss = 0.5 * delta**2 if direction == "reverse" else torch.exp(delta) - delta - 1
    return loss, value


def parse_strata(text):
    """'procedure:each=76,total=4' -> ('procedure', {'each': 76, 'total': 4})."""
    field, counts = text.split(":", 1)
    return field, {k: int(v) for k, v in (item.split("=") for item in counts.split(","))}


def draw_batch(rng, problems, prompts_per_step, strata=None):
    """Indices of one batch: uniform without replacement, or a fixed count from each stratum."""
    if strata is None:
        return rng.choice(len(problems), size=prompts_per_step, replace=False).tolist()
    field, counts = strata
    picked = []
    for value, count in counts.items():
        members = [i for i, q in enumerate(problems) if q[field] == value]
        picked += rng.choice(members, size=count, replace=False).tolist()
    return picked


def policy_loss(advantages, seq_logp, sampler_logp=None, tis=None):
    """-(w * A * log pi).sum(). With tis = C, w = min(pi / pi_sampler, C) corrects for the samples having
    come from vLLM's copy of the policy (bf16, possibly quantized), which can drift from the trainer's."""
    if tis is None:
        return -(advantages * seq_logp).sum()
    weights = torch.exp(seq_logp.detach() - sampler_logp).clamp(max=tis)
    return -(weights * advantages * seq_logp).sum()


def drift_stats(trainer_logp, sampler_logp, lengths):
    """Sampler-trainer mismatch over the sampled answers: per-token mean |gap|, per-token signed gap
    (trainer - sampler), and the largest per-sequence |gap|, in nats."""
    gap = np.asarray(trainer_logp, dtype=float) - np.asarray(sampler_logp, dtype=float)
    tokens = max(float(np.sum(lengths)), 1.0)
    return {"drift_abs_per_token": float(np.abs(gap).sum() / tokens), "drift_per_token": float(gap.sum() / tokens),
            "drift_max_seq": float(np.abs(gap).max()) if len(gap) else 0.0}


def completion_logprobs(model, sequences, prompt_lens, pad_id, ref=None, adapter_off=False, kl="reverse"):
    """Per-sequence sum of completion token log-probs, and (with ref) the summed KL loss term and value."""
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
        loss, value = kl_terms(logp, ref_logp, kl)
        return seq_logp, ((loss * token_mask).sum(dim=1), (value * token_mask).sum(dim=1))
    return seq_logp, None


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


def export(model, tokenizer, path, args):
    """Save the current policy for evaluate.py; training state is untouched.

    merged sync: a bf16 model with LoRA merged in. adapter sync: the adapter only
    (evaluate.py --adapter <path> --model <base> [--quant int8]), so evaluation
    uses the same quantized base the policy was trained and sampled on.
    """
    if args.sync == "adapter":
        model.save_pretrained(path)
    else:
        inner = model.base_model.model if args.lora else model
        inner.save_pretrained(path, state_dict=dict(vllm_weights(model, args.lora)))
    tokenizer.save_pretrained(path)


# ---------------------------------------------------------------- main

def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from vllm import SamplingParams

    problems = json.load(open(args.problems_file)) if args.problems_file else load_gsm8k(args.split)
    if args.problems:
        keep = set(json.load(open(args.problems)))
        problems = [p for p in problems if p["id"] in keep]
    model, tokenizer = build_policy(args)
    ref = reference_model(args, model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.lora:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    else:
        # 8-bit Adam state (1 GB instead of 4 GB for 0.5B) so training fits beside vLLM on 24 GB.
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sampler = Sampler(args)

    rng = np.random.default_rng(args.seed)
    step = 0
    ckpt = os.path.join(args.out, "checkpoint")
    if os.path.exists(os.path.join(ckpt, "state.pt")):
        step, rng_state = load_checkpoint(ckpt, model, optimizer, args.lora)
        rng.bit_generator.state = rng_state
        print(f"resumed at step {step}", flush=True)
    sampler.sync(model)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.check_sync:
        print(json.dumps({"step": step, "sync_check": sync_check(sampler, model, tokenizer, problems[0], pad, 0)}),
              flush=True)

    log_path = os.path.join(args.out, "log.jsonl")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    last_save = time.time()
    scale_path = os.path.join(args.out, "agg_scale.txt")
    agg_scale = float(open(scale_path).read()) if os.path.exists(scale_path) else None

    strata = parse_strata(args.strata) if args.strata else None
    while step < args.steps:
        t0 = time.time()
        batch = [problems[i] for i in draw_batch(rng, problems, args.prompts_per_step, strata)]
        prompts = [prompt_text(tokenizer, p) for p in batch]
        track = args.drift_log or args.tis is not None
        params = SamplingParams(n=args.G, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                                seed=int(rng.integers(2**31)), logprobs=0 if track else None)
        outputs = sampler.generate(prompts, params)
        t_sample = time.time() - t0

        rewards = np.zeros((len(batch), args.G))
        parsed = np.zeros_like(rewards)
        sequences, prompt_lens, lengths, sampler_logp = [], [], [], []
        for i, (problem, out) in enumerate(zip(batch, outputs)):
            for g, c in enumerate(out.outputs):
                rewards[i, g], parsed[i, g] = reward(c.text, problem["answer"])
                if track:
                    total = c.cumulative_logprob
                    if total is None:
                        total = sum(lp[t].logprob for lp, t in zip(c.logprobs, c.token_ids))
                    sampler_logp.append(float(total))
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
        n_total = len(sequences) if args.denominator is None else args.denominator * args.G
        norm = n_total
        if args.agg == "token":
            if agg_scale is None:
                agg_scale = float(np.mean(lengths))
                with open(os.path.join(args.out, "agg_scale.txt"), "w") as f:
                    f.write(f"{agg_scale}\n")
            norm = sum(lengths) / agg_scale
        optimizer.zero_grad(set_to_none=True)
        kl_sum = 0.0
        seen_logp = {}
        sampler_t = torch.tensor(sampler_logp, device="cuda") if track else None
        for start in range(0, len(active), args.micro_batch):
            idx = active[start: start + args.micro_batch]
            seq_logp, kl = completion_logprobs(model, [sequences[i] for i in idx], [prompt_lens[i] for i in idx],
                                               pad, ref=None if args.beta <= 0 else (model if args.lora else ref),
                                               adapter_off=bool(args.lora), kl=args.kl)
            loss = policy_loss(A[idx], seq_logp, None if args.tis is None else sampler_t[idx], args.tis) / norm
            if track:
                seen_logp.update(zip(idx, seq_logp.detach().tolist()))
            if kl is not None:
                kl_loss, kl_value = kl
                loss = loss + args.beta * kl_loss.sum() / n_total
                kl_sum += kl_value.sum().item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1e9)  # norm only, no clipping
        grad_norm = float(sum(p.grad.float().norm() ** 2 for p in model.parameters()
                              if p.requires_grad and p.grad is not None) ** 0.5)
        skipped = not np.isfinite(grad_norm)
        if skipped:
            # A non-finite gradient would destroy the weights; drop this update and keep going.
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            sampler.sync(model)
        step += 1

        group = clean.mean(axis=1)
        record = {"step": step, "reward": float(clean.mean()), "reward_used": float(rewards.mean()),
                  "informative": float(((group > 0) & (group < 1)).mean()), "parsed": float(parsed.mean()),
                  "length": float(np.mean(lengths)), "active": len(active), "grad_norm": grad_norm,
                  "kl": kl_sum / n_total, "skipped": int(skipped), "t_sample": t_sample, "t_step": time.time() - t0,
                  "peak_gb": torch.cuda.max_memory_allocated() / 2**30}
        if track and seen_logp:
            used = sorted(seen_logp)
            record.update(drift_stats([seen_logp[i] for i in used], [sampler_logp[i] for i in used],
                                      [lengths[i] for i in used]))
        if args.check_sync and step <= 3:
            record["sync_check"] = sync_check(sampler, model, tokenizer, problems[0], pad, 0)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        # Which problems were sampled and what each group earned: per-problem exposure and, for C2,
        # informative (mixed-reward) groups per procedure.
        with open(os.path.join(args.out, "groups.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, "ids": [q["id"] for q in batch],
                                "rewards": clean.astype(int).tolist()}) + "\n")
        print(json.dumps(record), flush=True)
        if step in args.save_at:
            export(model, tokenizer, os.path.join(args.work, f"step_{step}"), args)

        if time.time() - last_save > 60 * args.save_minutes or step == args.steps:
            save_checkpoint(ckpt, model, optimizer, step, rng.bit_generator.state, args.lora)
            last_save = time.time()

    final = os.path.join(args.work, "final")
    export(model, tokenizer, final, args)
    # The checkpoint (~3 GB for 0.5B) stays on the home disk until the job's evaluations are done, so a
    # job interrupted after training (03:00 /tmp wipe) resumes at the last step and only re-exports.
    # Job lines delete it at the end (llm/jobs/*.txt).
    print(f"saved {final}", flush=True)


if __name__ == "__main__":
    main()
