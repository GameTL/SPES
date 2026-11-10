# SPES reproduction report

Paper: *SPES: Towards Optimizing Performance-Resource Trade-Off for Serverless Functions* (arXiv 2403.17574, ICDE'24)

Data: Azure Functions Trace 2019, 14 days. Days 1–12 train, days 13–14 simulate.
Source: https://github.com/Azure/AzurePublicDataset/releases (dataset-functions-2019).

Run: `repro_run.py` (notebook `run.ipynb` converted to a script; loader vectorised, counts stored as `int32`).
Summary: `result/summary.json`.

## Headline comparison

| Metric | Paper | Notebook | Azure 2019 rerun | Azure 2021 | Huawei 2025 R5 |
|---|---|---|---|---|---|
| Functions simulated | 39,388 | 39,388 | 39,388 | 238 | 497 |
| Q3 cold-start rate (SPES) | 0.108 | 0.108 | **0.0958** | **1.0** | **0.0** |
| Median CSR | – | – | 0.0 | 0.1491 | 0.0 |
| P90 CSR | – | – | 0.6667 | 1.0 | 0.0 |
| Functions with zero cold starts | 57.99% | – | **58.64%** | **13.45%** | **91.15%** |
| Always-cold functions | < 8% | – | **7.63%** | **27.73%** | **0.60%** |
| Total cold starts | – | – | 218,394 | 988 | 709 |
| Total invocations (test days) | – | – | 1,866,622,093 | 229,345 | 7,633,975 |
| WMT (minutes, all functions) | – | 21,521,200 | **19,271,548** | 77,069 | 111,317 |

Azure 2019 rerun cells are the existing paper-comparison column (Q3 0.0958). That column was not recomputed. The later recovery in `result/summary.json` is Q3 0.0959.

Azure 2021 and Huawei 2025 R5 are **not** comparable to the paper or to the Azure 2019 rerun. Same simulator, different traces. Precomputed correlation is off: the candidate map is empty, and `mid-data/func_candi.json` (2019 hashes) was not read. Summaries: `result/summary_azure2021.json`, `result/summary_huawei.json`.

**Azure 2021.** 14 days from 2021-01-31 (1,980,951 events, 424 functions; 238 appear on the test days). Minute = floor(end_timestamp/60). Max minute count 3,212; nothing was clipped to int32 and no timestamp fell outside the 14×1440 grid. Owner and application are the app id. Function key is app+func. Trigger is `unknown`. No function is typed CORR. The script still links test-only functions that share an app (40 functions, 361 links) and those links stay for both test days. Q3 of 1.0 is the 75th percentile on this small sparse trace (66/238 functions have CSR 1).

**Huawei 2025 R5.** Region 5 only, quantile `requests` as the per-minute invocation count, not the five-region corpus. The series is 31 days; train is days 0–11 and test is days 12–13 so the 12+2 split matches SPES. 543 functions in the panel, 497 simulated. Max minute count 18,119; counts were already integers and none were clipped. Each funcID is its own owner and application, so there are no owner/app correlation links. Trigger is `unknown`. Q3 of 0.0 means at least three quarters of these functions had CSR 0 (453/497); there were still 709 cold starts. These are busy functions (up to ~18k requests in one minute), not the Azure 2019 long tail.

## Verdict

- Headline Q3-CSR: the Azure 2019 rerun gives 0.0958 vs the paper's 0.108 (about 11% lower). Same order of magnitude, same direction. The Azure 2021 and Huawei columns are a different setup; see the table notes.
- Zero-cold share (58.6% vs 57.99%) and always-cold share (7.6% vs under 8%) match the paper closely.
- WMT: 19.27M here vs 21.52M in the notebook's saved output (about 10% lower). The notebook's saved run is the closer comparison to this code.

## Known deviations from the paper (not changed in this run)

1. `PLUSED_GIVEUP = 3` in the notebook; the paper states θgivenup = 5 for "dense" and "plused".
2. `cal_corr.py` default `--warm_th 0.7` (paper uses T-COR threshold 0.5).
3. Online-correlation pruning (`test_func_corr_perform`) appears to never execute: the dict comprehension builds a single key, and the loop unpacks keys instead of items. Impact on Q3-CSR is probably small (paper reports it as a small effect).
4. Correlation step (cells 33–45) is skipped in `repro_run.py`; it relies on the shipped `mid-data/func_candi.json` (produced by `cal_corr.py`).
5. Random seed fixed to 0 here; the paper does not state one, so exact matches are not expected.

## Limitations of this run

- `func_invok_*.json` from the simulation was truncated when the process was killed during the final dump. Invocation totals were rebuilt from the test CSVs (`result/func_invok_rebuilt.json`); cold-start rates use these rebuilt counts and match the simulated denominators.
- Baselines (Defuse, FaaSCache, Hybrid, fixed keep-alive) are not in this repo, so the paper's 49.77% / 56.43% improvement claims were not reproduced here.
- The notebook's cells 74+ (plots) were not executed.
