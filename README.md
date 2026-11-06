# 981_project

Optimization Project for SFU CMPT981

## Shared synthetic experiment code

Use the components in `synthetic_rl/` when building a synthetic experiment:

- **Use `synthetic_task.py` to generate data and verify answers.**
- **Use `policy.py` to create the model that learns to answer.**
- **Use `trainer.py` to train and evaluate the model.** Create a `Trainer` with the policy, optimizer, and training settings, then call `train()`. It handles sampling, verification, advantage calculation, and policy updates. Use `evaluate()` to measure performance without updating the model.
