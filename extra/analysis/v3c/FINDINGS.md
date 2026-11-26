# v3c: pass@k training rules (exploratory, PREREG_v3c.md)

1920 runs. Each cell: mean over seeds of (rule − grpo) at the same (level, G, η); one-sided Wilcoxon over paired seeds. Exact pass@k from the final p.

## 1. Minority task: share of minority prompts absorbed (prediction: pass@k rules lower)

| rule | cells_lower | mean_diff |
|---|---|---|
| mean | 2/6 | -0.0158 |
| passk2 | 6/6 | -0.98 |
| passk4 | 6/6 | -1 |
| passk8 | 6/6 | -1 |
| rloo_k2 | 6/6 | -0.949 |
| rloo_k4 | 6/6 | -1 |
| rloo_k8 | 6/6 | -1 |

## 2. Shared levels (lin, m64): pass@64 higher and pass@1 lower than grpo?

pass@64:

| rule | cells_higher | mean_diff |
|---|---|---|
| mean | 11/12 | 0.0994 |
| passk2 | 5/12 | 0.08 |
| passk4 | 7/12 | 0.155 |
| passk8 | 7/12 | 0.152 |
| rloo_k2 | 12/12 | 0.145 |
| rloo_k4 | 12/12 | 0.195 |
| rloo_k8 | 12/12 | 0.199 |

pass@1:

| rule | cells_lower | mean_diff |
|---|---|---|
| mean | 0/12 | 0.0988 |
| passk2 | 1/12 | 0.0792 |
| passk4 | 4/12 | 0.142 |
| passk8 | 5/12 | 0.0875 |
| rloo_k2 | 1/12 | 0.144 |
| rloo_k4 | 1/12 | 0.177 |
| rloo_k8 | 2/12 | 0.126 |

## 3. Tabular: absorption by rule (prediction: none)

| rule | max_absorbed |
|---|---|
| grpo | 0 |
| mean | 0 |
| passk2 | 0 |
| passk4 | 0 |
| passk8 | 0 |
| rloo_k2 | 0 |
| rloo_k4 | 0 |
| rloo_k8 | 0 |

## 4. Means by level and rule (over G, η, seeds)

| level | rule | pass1 | pass64 | absorbed |
|---|---|---|---|---|
| lin | grpo | 0.861 | 0.863 | 0.105 |
| lin | mean | 0.907 | 0.91 | 0.0624 |
| lin | passk2 | 0.853 | 0.856 | 0.104 |
| lin | passk4 | 0.853 | 0.857 | 0.104 |
| lin | passk8 | 0.846 | 0.852 | 0.106 |
| lin | rloo_k2 | 0.908 | 0.912 | 0.056 |
| lin | rloo_k4 | 0.93 | 0.937 | 0.0302 |
| lin | rloo_k8 | 0.932 | 0.944 | 0.0183 |
| m64 | grpo | 0.684 | 0.684 | 0.282 |
| m64 | mean | 0.836 | 0.836 | 0.143 |
| m64 | passk2 | 0.85 | 0.851 | 0.125 |
| m64 | passk4 | 0.976 | 1 | 3.81e-05 |
| m64 | passk8 | 0.874 | 1 | 0 |
| m64 | rloo_k2 | 0.924 | 0.926 | 0.0697 |
| m64 | rloo_k4 | 0.969 | 1 | 0 |
| m64 | rloo_k8 | 0.864 | 1 | 0 |
| minority | grpo | 0.897 | 0.897 | 0.103 |
| minority | mean | 0.897 | 0.898 | 0.101 |
| minority | passk2 | 0.894 | 0.995 | 0.00192 |
| minority | passk4 | 0.78 | 1 | 0 |
| minority | passk8 | 0.6 | 1 | 0 |
| minority | rloo_k2 | 0.88 | 0.986 | 0.00495 |
| minority | rloo_k4 | 0.748 | 1 | 0 |
| minority | rloo_k8 | 0.56 | 1 | 0 |
| tab | grpo | 0.997 | 1 | 0 |
| tab | mean | 0.965 | 1 | 0 |
| tab | passk2 | 0.963 | 1 | 0 |
| tab | passk4 | 0.854 | 1 | 0 |
| tab | passk8 | 0.721 | 1 | 0 |
| tab | rloo_k2 | 0.919 | 1 | 0 |
| tab | rloo_k4 | 0.799 | 1 | 0 |
| tab | rloo_k8 | 0.647 | 1 | 0 |
