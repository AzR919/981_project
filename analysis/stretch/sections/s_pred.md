## H3 stretch: per-prompt absorption predictor (800 runs)

Within-run AUC for which prompts get absorbed. Scores: −d_x (predicted first-order drift), −d_x/(p(1−p)), −p_x, −‖∇p_x‖, and a 5-fold cross-validated logistic model on (log p_x, drift). *Step 0*: absorbed at any later time. *Checkpoints*: absorbed within the next 500 steps, averaged over checkpoints. *sft_mixed* starts give a real spread of p; uniform starts give every prompt the same p, so the p baseline is 0.5 there by construction.

Median spread (sd) of starting p by level and init: c05: sft_mixed 0.12, uniform 0.00; c09: sft_mixed 0.09, uniform 0.00; lin: sft_mixed 0.15, uniform 0.00; m256: sft_mixed 0.37, uniform 0.00; m64: sft_mixed 0.30, uniform 0.00

Step-0 AUCs (median over runs):

| level | init | runs_scored | step0_drift | step0_logit_drift | step0_p | step0_grad | step0_combo |
|---|---|---|---|---|---|---|---|
| lin | sft_mixed | 80 | 0.729 | 0.756 | 0.520 | 0.536 | 0.760 |
| c05 | sft_mixed | 80 | 0.897 | 0.898 | 0.686 | 0.697 | 0.894 |
| c09 | sft_mixed | 46 | 0.999 | 0.997 | 0.861 | 0.854 | 0.996 |
| m64 | sft_mixed | 62 | 0.633 | 0.666 | 0.495 | 0.517 | 0.645 |
| m256 | sft_mixed | 60 | 0.629 | 0.672 | 0.511 | 0.518 | 0.656 |
| lin | uniform | 80 | 0.764 | 0.764 | 0.500 | 0.501 | 0.757 |
| c05 | uniform | 80 | 0.870 | 0.870 | 0.500 | 0.499 | 0.868 |
| c09 | uniform | 62 | 1.000 | 1.000 | 0.500 | 0.509 | 1.000 |
| m64 | uniform | 65 | 0.620 | 0.620 | 0.500 | 0.495 | 0.632 |
| m256 | uniform | 58 | 0.609 | 0.609 | 0.500 | 0.480 | 0.645 |

Checkpoint AUCs (absorbed within the next 500 steps; median over runs):

| level | init | ck_drift | ck_logit_drift | ck_p | ck_grad | ck_combo |
|---|---|---|---|---|---|---|
| lin | sft_mixed | 0.509 | 0.509 | 0.506 | 0.474 | 0.458 |
| c05 | sft_mixed | 0.517 | 0.527 | 0.658 | 0.230 | 0.508 |
| c09 | sft_mixed | 0.893 | 0.989 | 1.000 | 0.996 | 0.999 |
| m64 | sft_mixed | 0.502 | 0.502 | 0.496 | 0.497 | 0.463 |
| m256 | sft_mixed | 0.498 | 0.498 | 0.498 | 0.491 | 0.466 |
| lin | uniform | 0.496 | 0.496 | 0.499 | 0.473 | 0.465 |
| c05 | uniform | 0.509 | 0.511 | 0.558 | 0.258 | 0.513 |
| c09 | uniform | 0.825 | 1.000 | 1.000 | 0.932 | 0.999 |
| m64 | uniform | 0.502 | 0.502 | 0.499 | 0.498 | 0.476 |
| m256 | uniform | 0.505 | 0.505 | 0.505 | 0.478 | 0.476 |

| level | best_baseline | runs | drift_minus_baseline | ci_lo | ci_hi | ck_minus_step0 | ck_ci_lo | ck_ci_hi |
|---|---|---|---|---|---|---|---|---|
| lin | grad | 80 | 0.189 | 0.159 | 0.206 | -0.256 | -0.266 | -0.216 |
| c05 | grad | 80 | 0.172 | 0.153 | 0.205 | -0.289 | -0.415 | -0.16 |
| c09 | p | 46 | 0.136 | 0.0559 | 0.145 | -0.105 | -0.213 | -0.0402 |
| m64 | grad | 62 | 0.107 | 0.0686 | 0.157 | -0.153 | -0.199 | -0.108 |
| m256 | grad | 60 | 0.0963 | 0.0404 | 0.141 | -0.154 | -0.227 | -0.122 |

![step-0 AUC by score and level (sft_mixed starts)](figures/s_pred_auc.png)

**Verdict: PASS** (drift beats the best single baseline by ≥ 0.05 AUC with a bootstrap CI above 0 on c05 and c09)

**Verdict: FAIL** (on the MLPs step-0 drift does not beat the best baseline by 0.05)

**Verdict: FAIL** (on the MLPs checkpoint drift predicts better than step-0 drift (bootstrap CI above 0))

