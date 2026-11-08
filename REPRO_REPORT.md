# SPES reproduction report

Paper: "SPES: Towards Optimizing Performance-Resource Trade-Off for Serverless Functions" (ICDE'24, arXiv 2403.17574)
Data: Azure Functions 2019 trace (official release, 14 days; 12 train, 2 simulate)

## Headline metrics

| Metric | Paper | Notebook saved output (run.ipynb) | Our rerun (repro_run.py) |
|---|---|---|---|
| Q3 cold-start rate (SPES) | 0.108 | 0.108 | **0.0958** |
| Median CSR | not reported | not reported | 0.0 |
| P90 CSR | not reported | not reported | 0.6667 |
| Functions with 0 cold starts | 57.99% | not reported | **58.64%** |
| Always-cold functions | < 8% | not reported | **7.63%** |
| Functions simulated | 39,388 | 39,388 | 39,388 |
| Train functions | 71,616 | 71,616 | 71,616 |
| Test-only (never in train) | 743 | not reported | not reported |

## Categorization (train, before assignment)

| Type | Notebook | Our rerun |
|---|---|---|
| Warm | 1119 + 720 (almost + always) | matched |
| Regular | 16,193 | matched |
| Appro-regular | 6,242 | matched |
| Dense | 405 | matched |
| Successive | 1,683 | matched |
| Forgetting day 1..6 counts | 169, 171, 218, 230, 269, 3 | 169, 171, 218, 230, 269, 3 |
| Forgetting regular (1,2) | 92, 80 | 97, 86 |

## Indeterminate assignment

| Quantity | Notebook | Our rerun |
|---|---|---|
| Assigned to pulsed/possible/corr | 33,971 | 33,882 |
| Unknown after assignment | 10,223 | 10,223 |
| Tried for the three rules (plused, possible, corr) | 30,917 / 22,148 / 7,077 | 30,834 / 22,067 / 17,923 |
| Assigned forcibly (plused / possible / corr) | 844 / 7,809 / 1,769 | 857 / 7,488 / 1,998 |

## Discussion

- **Cold-start rate:** our rerun gives Q3 = 0.0958. The paper reports 0.108 and the notebook's saved output also shows 0.108. Our number is about 11% lower than both. Zero-cold share (58.64%) is close to the paper's 57.99%.
- **Always-cold share:** 7.63% is consistent with the paper's "just less than 8%".
- **Corr candidates differ:** the notebook tries 7,077 functions against the correlated rule; our run tries 17,923. The correlation input (func_candi.json) was shipped in mid-data, so this difference is not explained by thresholds I checked. I have not confirmed it is the cause of the Q3 gap. This is an open question.
- **Forgetting regular counts drift slightly** (92/80 vs 97/86). Cause not yet investigated.
- **Not verified:** baselines (FaaSCache, Defuse, Hybrid, Fixed-10min). The notebook only simulates SPES, so the 49.77% Q3 improvement over Defuse and the memory numbers were not reproduced here.

## Known deviations from the paper (code review)

1. PLUSED give-up is 3 in the notebook; the paper says 5.
2. cal_corr.py defaults to warm_th 0.7 (paper: 0.5).
3. The correlation pruning dict in cell 56 builds only one entry, so the online-correlation pruning may not run.
4. Count arrays are int32 (not Python lists), to fit 16 GB RAM; counts reach 181,302 per minute, so uint16 would overflow.

## Environment

- macOS 16 GB RAM, Python 3.12, numpy 2.5.4, pandas 3.1.1, scipy, matplotlib (Agg), tqdm.
- Run via tmux, about 75 minutes end to end.
