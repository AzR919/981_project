# H1 advantage-gap check

## What me did

1. Start each rule and seed from a fresh tabular policy with four zero logits. Each answer initially has probability $1/4$.
2. Before each SGD update, compute $p$ and its exact gradient with PyTorch autograd. Multiply the gradient by the rule's finite-group advantage weight to get the prediction.
3. Sample 512 groups of four answers from that same policy and take one SGD step using the selected rule. The gradient used by SGD gives the sampled side of the comparison. Both gradients describe the policy at the start of the step.
4. Square each gradient norm, then average the squared norms across the 10 seeds at each step. Plot the predicted norm as a dashed line and the sampled norm as a solid line on the same log-scale axis.

## Fixed settings

| Setting | Value |
| --- | --- |
| Policy | Bias-free linear tabular policy with one prompt and four logits |
| Response length and correct answer | One token, answer ID `0` |
| Initial success probability $p_0$ | $0.25$ |
| Responses per group $G$ | $4$ |
| Groups per SGD step | $512$ |
| Sampled responses per step | $2{,}048$ |
| Optimizer and learning rate | Plain SGD with fixed learning rate $0.4$ for all three rules |
| SGD steps per run | $250$ |
| Sampling seeds | $0$ through $9$ for each rule |
| KL coefficient $\beta$ | $0$ |
| GRPO standardization constant $\epsilon$ | $10^{-5}$ |
