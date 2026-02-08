# Consistency Check

This report flags potential *size/metric mismatches* (e.g., prune-only size unexpectedly close to unpruned).

| Scene | Exp | Method | PSNR | SSIM | LPIPS | #G | MB | Flag |
|---|---|---|---:|---:|---:|---:|---:|---|
| bicycle | baseline_bicycle_i15k_noprune | baseline | NA | NA | | | 4728473 | 1118.34 | FLAG_TOO_LARGE |
| bonsai | baseline_bonsai_i15k_noprune | baseline | NA | NA | | | 1066779 | 252.31 | OK |
| garden | baseline_garden_i15k_noprune | baseline | NA | NA | | | 4210091 | 995.74 | FLAG_TOO_LARGE |
| kitchen | baseline_kitchen_i15k_noprune | baseline | NA | NA | | | 1554339 | 367.62 | OK |
| bicycle | gated_bicycle_i15k_tau30 | gated+pruned | NA | NA | | | 1187330 | 280.82 | OK |
| bonsai | gated_bonsai_i15k_tau30 | gated+pruned | NA | NA | | | 501857 | 118.7 | OK |
| garden | gated_garden_i15k_nogateloss | unknown | NA | NA | | | 2031532 | 480.48 | OK |
| garden | gated_garden_i15k_noprune | gated(no-prune) | NA | NA | | | 4190053 | 991.0 | FLAG_TOO_LARGE |
| garden | gated_garden_i15k_tau20 | gated+pruned | NA | NA | | | 2093227 | 495.07 | OK |
| garden | gated_garden_i15k_tau30 | gated+pruned | NA | NA | | | 1909043 | 451.51 | OK |
| garden | gated_garden_i15k_tau40 | gated+pruned | NA | NA | | | 2027533 | 479.54 | OK |
| garden | gated_garden_i15k_tau50 | gated+pruned | NA | NA | | | 2021158 | 478.03 | OK |
| kitchen | gated_kitchen_i15k_tau30 | gated+pruned | NA | NA | | | 364682 | 86.25 | OK |

| bicycle | pruneonly_bicycle_i15k_tau30 | prune-only | NA | NA | | | 2134976 | 504.95 | OK |

| bonsai | pruneonly_bonsai_i15k_tau30 | prune-only | NA | NA | | | 478355 | 113.14 | OK |

| garden | pruneonly_garden_i15k_tau30 | prune-only | NA | NA | | | 1885615 | 445.97 | OK |

| kitchen | pruneonly_kitchen_i15k_tau30 | prune-only | NA | NA | | | 687757 | 162.66 | OK |
| garden | random_garden_i15k_keep45 | random | NA | NA | | | 1896393 | 448.52 | OK |
