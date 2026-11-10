# SPES reproduction — Tinapat (other machine)

Paper: [SPES (arXiv 2403.17574)](https://arxiv.org/abs/2403.17574) — cold-start scheduling on Azure Functions traces.

This repo: `git clone` [https://github.com/BEbillionaireUSD/SPES](https://github.com/BEbillionaireUSD/SPES) (or your fork).  
`raw_data/` **is gitignored** — download datasets below after clone.

Suggested RAM: **32 GB+** for full Azure 2019 pipeline; **CPU-only** (no GPU).

---

## 1. Azure Functions 2019 (SPES default)

**What SPES expects:** 14 daily CSVs  
`invocations_per_function_md.anon.d01.csv` … `d14.csv`

| Item                      | URL                                                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Archive (recommended)** | https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz |
| Release page              | https://github.com/Azure/AzurePublicDataset/releases/tag/dataset-functions-2019                                                                   |
| Dataset docs              | https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md                                                              |

**Install path**

```text
SPES/raw_data/azurefunctions-dataset2019/
  invocations_per_function_md.anon.d01.csv
  …
  invocations_per_function_md.anon.d14.csv
```

**Extract (invocation files only; ~1.8 GB uncompressed)**

```bash
cd SPES
mkdir -p raw_data/azurefunctions-dataset2019
cd raw_data
curl -L -o azure2019.tar.xz \
  "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
tar -xJf azure2019.tar.xz 'invocations_per_function_md.anon.d*.csv' -C azurefunctions-dataset2019 --strip-components=0
# If files land in CWD, move them into azurefunctions-dataset2019/
```

**Run (this fork)**

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scipy matplotlib tqdm
# long run (~1–2 h CPU); use tmux/screen
python repro_run.py
# metrics: result/summary.json
```

---

## 2. Azure Functions 2021 (newer Azure; optional comparison)

Event-level trace (not native SPES format). Bucket into per-minute matrices before training.

| Item | URL |
|------|-----|
| **Download (.rar)** | https://github.com/Azure/AzurePublicDataset/raw/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar |
| Docs | https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md |

```bash
mkdir -p raw_data/azurefunctions-invocation-2021
# needs unrar or 7z
curl -L -o raw_data/azurefunctions-invocation-2021/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar \
  "https://github.com/Azure/AzurePublicDataset/raw/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar"
```

Convert, then run the same simulator with an empty candidate map (does not read `mid-data/func_candi.json`):

```bash
python convert_traces.py
python repro_dataset.py azure2021
# metrics: result/summary_azure2021.json
```

---

## 3. Huawei Public FaaS trace 2025 (comparison / newer production data)

Paper: *[Serverless Cold Starts and Where to Find Them](https://arxiv.org/abs/2410.06145)* (EuroSys 2025).  
Upstream: https://github.com/sir-lab/data-release/blob/master/README_data_release_2025.md

**All files are on Google Drive.** Install `gdown`: `pip install gdown`.

```bash
mkdir -p raw_data/huawei-2025
cd raw_data/huawei-2025
```

### 3a. Cold start events (31 days, per region)

| Region | Google Drive (browser) | `gdown` ID |
|--------|------------------------|------------|
| R1 | https://drive.google.com/file/d/1mMQtfZNtg-EPmGmGYuOzC5KPbZoXRd8e/view | `1mMQtfZNtg-EPmGmGYuOzC5KPbZoXRd8e` |
| R2 | https://drive.google.com/file/d/1qjnBBH_yEOnxAG5vFPQwx6xyITZrrP8L/view | `1qjnBBH_yEOnxAG5vFPQwx6xyITZrrP8L` |
| R3 | https://drive.google.com/file/d/1WlUk-tQLao6mU0FoMx9YO2-HIMevEDjX/view | `1WlUk-tQLao6mU0FoMx9YO2-HIMevEDjX` |
| R4 | https://drive.google.com/file/d/1YywoClXTZ1Q0jKfB2Y3Hgu0xh0lrDoBZ/view | `1YywoClXTZ1Q0jKfB2Y3Hgu0xh0lrDoBZ` |
| R5 | https://drive.google.com/file/d/1ah7nj5fyedMpOeaFgNVA7oMx3815D8BZ/view | `1ah7nj5fyedMpOeaFgNVA7oMx3815D8BZ` |

```bash
gdown 1ah7nj5fyedMpOeaFgNVA7oMx3815D8BZ -O region5_coldstarts.zip
```

### 3b. Per-minute time series quantiles (31 days, per region; large ~0.9 GB each)

Best source for **invocation rate** comparison (`requests`, `num_cold_starts`, etc. inside each zip).

| Region | Google Drive | `gdown` ID |
|--------|--------------|------------|
| R1 | https://drive.google.com/file/d/1XMSCDTRSEhJRSHA2gLuRM_WEYuuXVyWJ/view | `1XMSCDTRSEhJRSHA2gLuRM_WEYuuXVyWJ` |
| R2 | https://drive.google.com/file/d/1iZk-n_5IT7j3M_pHpRzhwSPTuJ2vxvlR/view | `1iZk-n_5IT7j3M_pHpRzhwSPTuJ2vxvlR` |
| R3 | https://drive.google.com/file/d/1xDPmacasZqVB_lM5UwJSiRKRNjg-ImvD/view | `1xDPmacasZqVB_lM5UwJSiRKRNjg-ImvD` |
| R4 | https://drive.google.com/file/d/1KNDZ1rHEvcfdp92xklmJFlLUPxFbQ75N/view | `1KNDZ1rHEvcfdp92xklmJFlLUPxFbQ75N` |
| R5 | https://drive.google.com/file/d/1BxLz8ETUHZXbxr8PkDRFDwnBjLtCfidR/view | `1BxLz8ETUHZXbxr8PkDRFDwnBjLtCfidR` |

```bash
gdown 1BxLz8ETUHZXbxr8PkDRFDwnBjLtCfidR -O region5_quantiles.zip
```

### 3c. Trigger types and runtimes (Region 2 only; small)

| Google Drive | `gdown` ID |
|--------------|------------|
| https://drive.google.com/file/d/1dUg_yxeLR5OyldivPZGW5djYF31166EG/view | `1dUg_yxeLR5OyldivPZGW5djYF31166EG` |

```bash
gdown 1dUg_yxeLR5OyldivPZGW5djYF31166EG -O region2_runtimes_triggers.csv
```

(File may be CSV despite `.zip` name in some mirrors.)

### 3d. Request tables (day 30, event-level; very large)

Shard links (20 parts for R1, etc.) are listed only in the upstream README — download selectively if you need raw events:

https://github.com/sir-lab/data-release/blob/master/README_data_release_2025.md#huawei-public-request-tables

**Huawei vs SPES:** `convert_traces.py` reads Region 5 quantile `requests` only (days 0–11 train, 12–13 test of 31) and writes `raw_data/huawei2025-spes/`. Then `python repro_dataset.py huawei` → `result/summary_huawei.json`. Column label is **Huawei 2025 R5**, not the whole corpus.

---

## 4. Side-by-side comparison (what to run where)

| Dataset | Role | In-repo runner today |
|---------|------|----------------------|
| **Azure 2019** | Reproduce SPES paper metrics | `repro_run.py` |
| **Azure 2021** | Newer Azure invocations | `convert_traces.py` then `repro_dataset.py azure2021` |
| **Huawei 2025 R5** | Region 5 quantile `requests` | `convert_traces.py` then `repro_dataset.py huawei` |

Same headline table as `REPRODUCTION.md`. Azure 2021 and Huawei 2025 R5 use an empty candidate map (no `cal_corr`). They are not paper-comparable.

| Metric | Paper | Notebook | Azure 2019 rerun | Azure 2021 | Huawei 2025 R5 |
|---|---|---|---|---|---|
| Functions simulated | 39,388 | 39,388 | 39,388 | 238 | 497 |
| Q3 cold-start rate (SPES) | 0.108 | 0.108 | **0.0958** | **1.0** | **0.0** |
| Functions with zero cold starts | 57.99% | – | **58.64%** | **13.45%** | **91.15%** |
| Always-cold functions | < 8% | – | **7.63%** | **27.73%** | **0.60%** |

Azure 2019 rerun is the existing column (not recomputed). Huawei train/test is days 0–11 / 12–13 of a 31-day Region 5 series. Notes and the other measured rows (median, P90, WMT, totals): `REPRODUCTION.md`. Summaries: `result/summary_azure2021.json`, `result/summary_huawei.json`.

---

## 5. Correlation artifact (Azure only)

`repro_run.py` reads `mid-data/func_candi.json` (~25 MB). The shipped file is enough to reproduce without recomputing correlation.

To recompute, first produce the categorization inputs (notebook cells that write these):

```text
mid-data/cal_corr/to_cal_corr.txt
mid-data/cal_corr/unknown_ownerapp_lst.txt
```

Then run shards **one after another** (not parallel). Each shard is the same `cal_corr.run_shard` rule (`warm_th=0.7`, lags 0–10) and is saved as its own file. After the last shard, the script reads each file, reads them all again, compares the two reads, writes a separate merged JSON (does not replace `func_candi.json`), and saves a bar chart.

```bash
# from the repo root, after mid-data/cal_corr/ exists
python eval_corr_seq.py
# shards:  mid-data/func_candi_tuple_lst_seq/0.txt, 1.txt, ...
# compare: result/corr_seq_compare.json
# graph:   result/corr_seq_compare.png
# merged:  mid-data/func_candi_seq.json
```

`python eval_corr_seq.py --self-check` runs the same read / re-read / compare / graph path on a tiny synthetic series (no Azure dump).

A single shard, same algorithm, still works if you only want one slice:

```bash
python cal_corr.py --start 0 --total 3704 \
  --corr-dir mid-data/cal_corr \
  --out-dir mid-data/func_candi_tuple_lst_seq
```

---

## 6. Checklist after clone

- [ ] `raw_data/azurefunctions-dataset2019/*.csv` (14 files)
- [ ] (Optional) `raw_data/huawei-2025/` downloads for comparison
- [ ] `mid-data/func_candi.json` present in repo (or `mid-data/func_candi_seq.json` after `eval_corr_seq.py`)
- [ ] `python repro_run.py` → `result/summary.json`

**Citations:** Azure 2019 → ATC’20 “Serverless in the Wild”; Azure 2021 → SOSP’21 harvested resources; Huawei 2025 → EuroSys’25 cold starts paper (links in upstream README).
