# Limitations (draft for the final report)

1. **Model size.**
   - We train models of 0.5B–1.7B (probes up to 3B). 7B/8B do not fit on one RTX 4090 with in-process
     sampling: vLLM's bf16 copy (about 15 GB) plus the training state is over 24 GB.
   - vLLM 0.34 has no bitsandbytes 4-bit path, and GSM8K is nearly solved at that size (Qwen2.5-7B base
     p̂ 0.96).
   - Llama-3.1-8B also needs a separate licence.
   - Most RLVR papers in our notes use 7B–8B (20 and 16 of 75 notes).
2. **Binary rewards only.** Every result assumes a deterministic 0/1 verifier, apart from the
   deliberately flipped-verifier ablations. The exact-gradient collapse of Huang et al. (2611.14467)
   needs graded rewards and is not covered. Reference resets were dropped for this reason.
3. **No autoregressive structure in the toy.** The L answer tokens are independent given the prompt.
   Length effects (Suk & Duan's 1/T) are therefore only partly represented: the toy has fixed L, and
   token aggregation is tested only in the LLM (C9).
4. **Cut LLM experiments.** The LLM drift predictor (P3 transferred) and the LLM group-size test were
   cut. P2 (kick size against noise) and P3 are toy-only.
5. **Family differences are confounded with the training method.** Qwen-0.5B is trained with full
   fine-tuning in fp32 with 8-bit Adam; SmolLM2 and Llama with LoRA and AdamW, at different learning
   rates. A cross-family difference cannot be attributed to the family alone.
6. **Sampler–trainer mismatch.** vLLM samples in bf16 while the learner computes log-probs from fp32
   master weights (full fine-tuning) or a merged LoRA. There is no importance correction. The SmolLM2
   pilots collapsed after about 100 steps at every learning rate but the highest (see HANDOFF 3g).
7. **GSM8K contamination.** Qwen's base p̂ is 0.42 on train problems against 0.32 on test. "Held-out"
   C1 problems are other train problems, so test-set results are reported separately.
