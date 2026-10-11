# Complete group-size evidence

Difficulty proxy: mean reward across all eight completions per prompt; at or below the frozen median (2.19873046875) is harder, above it is easier. Each bin has 12 prompts and 96 generations per K. The overall budget is 192.

| K | Bin | Groups | Informative % | Mean group reward std | Relative-signal variance |
|---:|---|---:|---:|---:|---:|
| 2 | overall | 96 | 93.75 | 0.384567 | 0.937481 |
| 2 | harder_proxy | 48 | 89.58 | 0.528330 | 0.895826 |
| 2 | easier_proxy | 48 | 97.92 | 0.240804 | 0.979136 |
| 4 | overall | 48 | 95.83 | 0.541004 | 0.958328 |
| 4 | harder_proxy | 24 | 91.67 | 0.711506 | 0.916663 |
| 4 | easier_proxy | 24 | 100.00 | 0.370502 | 0.999992 |
| 8 | overall | 24 | 95.83 | 0.619054 | 0.958329 |
| 8 | harder_proxy | 12 | 91.67 | 0.800249 | 0.916664 |
| 8 | easier_proxy | 12 | 100.00 | 0.437860 | 0.999994 |

This is one fixed partition per K, not a repeated-partition uncertainty estimate. The reward-derived bins are proxies, not independent difficulty labels. Relative-signal variance partly follows informative-group rate by construction.
