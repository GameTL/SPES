#!/usr/bin/env python3
"""Run repro_run.py categorization + simulation on a converted trace.

Does not import repro_run.py (import would start the Azure 2019 run).
Does not read or write mid-data/func_candi.json.
Does not write mid-data/cal_corr/ (eval_corr_seq.py uses that directory).
Candidate map is empty: no cross-function correlation from cal_corr.
"""

import json
import os
import sys
import traceback
from collections import defaultdict

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import random  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "azure2021": {
        "data_dir": "raw_data/azurefunctions-2021-spes",
        "summary": "result/summary_azure2021.json",
        "column": "Azure 2021",
    },
    "huawei": {
        "data_dir": "raw_data/huawei2025-spes",
        "summary": "result/summary_huawei.json",
        "column": "Huawei 2025 R5",
    },
    "globus": {
        "data_dir": "raw_data/globus-half-spes",
        "summary": "result/summary_globus_half.json",
        "column": "Globus half",
    },
    "ibm": {
        "data_dir": "raw_data/ibm-spes",
        "summary": "result/summary_ibm.json",
        "column": "IBM Code Engine",
    },
    "huawei_all": {
        "data_dir": "raw_data/huawei2025-all-spes",
        "summary": "result/summary_huawei_all.json",
        "column": "Huawei 2025 all regions",
    },
    "huawei_inflated": {
        "data_dir": "raw_data/huawei2025-all-spes",
        "summary": "result/summary_huawei_inflated.json",
        "column": "Huawei 2025 all regions inflated",
        "inflate_to": 39388,
    },
    "azure2021_inflated": {
        "data_dir": "raw_data/azurefunctions-2021-spes",
        "summary": "result/summary_azure2021_inflated.json",
        "column": "Azure 2021 inflated",
        "inflate_to": 39388,
    },
}


def load_split(files, ndays):
    # Same loader as repro_run.py (Azure 2019 path is not used here).
    arrcount, owner_app, owner_func, app_func, trig = {}, {}, defaultdict(set), defaultdict(set), defaultdict(set)
    for i, file in enumerate(files):
        df = pd.read_csv(file)
        counts = df.iloc[:, 4:].to_numpy(dtype=np.int32)
        funcs = df["HashFunction"].tolist()
        owners = df["HashOwner"].tolist()
        apps = df["HashApp"].tolist()
        trigs = df["Trigger"].tolist()
        for r, func in enumerate(funcs):
            owner_app[func] = owners[r] + "\t" + apps[r]
            owner_func[owners[r]].add(func)
            app_func[apps[r]].add(func)
            trig[func].add(trigs[r])
            if func not in arrcount:
                arrcount[func] = np.zeros(ndays * 1440, dtype=np.int32)
            arrcount[func][i * 1440:(i + 1) * 1440] = counts[r]
        del df, counts
    return arrcount, owner_app, owner_func, app_func, trig


def algorithm_source(tag):
    path = os.path.join(ROOT, "repro_run.py")
    text = open(path, encoding="utf-8").read()
    marker = "# ---- cell 2 ----"
    src = text[text.index(marker):]
    old_candi = 'train_candi = read_json("mid-data/func_candi.json")'
    if old_candi not in src:
        raise SystemExit("repro_run.py no longer reads func_candi.json where expected")
    src = src.replace(
        old_candi,
        "train_candi = {}  # empty map; the 2019 candidate file is not used",
        1,
    )
    run_dir = "result/runs/%s" % tag
    src = src.replace(
        'os.makedirs("mid-data/cal_corr", exist_ok=True)',
        'os.makedirs("%s", exist_ok=True)' % run_dir,
    )
    src = src.replace("mid-data/train_info.txt", run_dir + "/train_info.txt")
    src = src.replace("mid-data/cal_corr/to_cal_corr.txt", run_dir + "/to_cal_corr.txt")
    src = src.replace(
        "mid-data/cal_corr/unknown_ownerapp_lst.txt",
        run_dir + "/unknown_ownerapp_lst.txt",
    )
    src = src.replace("mid-data/train_info_assigned.txt", run_dir + "/train_info_assigned.txt")
    old_dump = '''cur_time = time.strftime("%d-%H-%M", time.localtime())

json_pretty_dump(func_cold, f"result/func_cold_{cur_time}.json")
json_pretty_dump(func_waste, f"result/func_waste_{cur_time}.json")
json_pretty_dump(func_invok, f"result/func_invok_{cur_time}.json")
json_pretty_dump(memory_size, f"result/memory_size_{cur_time}.json")
'''
    new_dump = '''os.makedirs("%s", exist_ok=True)
json_pretty_dump(func_cold, "%s/func_cold.json")
json_pretty_dump(func_waste, "%s/func_waste.json")
json_pretty_dump(func_invok, "%s/func_invok.json")
json_pretty_dump(memory_size, "%s/memory_size.json")
''' % (run_dir, run_dir, run_dir, run_dir, run_dir)
    if old_dump not in src:
        raise SystemExit("repro_run.py dump block not found")
    src = src.replace(old_dump, new_dump, 1)
    old_perform = "test_func_corr_perform = {func: {candi_func: 0} for candi_func in test_func_corr[func]}"
    if old_perform not in src:
        raise SystemExit("test_func_corr_perform line not found")
    src = src.replace(
        old_perform,
        old_perform + """
# A non-empty dict aborts the minute loop: it iterates keys and unpacks them.
# The Azure 2019 run only finishes when this dict is empty, so pruning never runs.
# test_func_corr links stay; this does not repair the pruner.
try:
    for _perform_k, _perform_v in test_func_corr_perform:
        break
except ValueError:
    test_func_corr_perform = {}
    perform_pruning_disabled = True
else:
    perform_pruning_disabled = False
""",
        1,
    )
    src = src.replace("plt.show()", "plt.close('all')")
    # Minute counts are int32. Summing them with Python sum() stays in int32 and
    # wraps on this trace (max minute 2.8e6). Thresholds must see the full total.
    # Same integer when the total already fits in int32.
    int64_sums = [
        ("sum(lst[pre_pos:i+1])", "int(np.sum(lst[pre_pos:i+1], dtype=np.int64))"),
        ("sum(lst[pre_pos:i])", "int(np.sum(lst[pre_pos:i], dtype=np.int64))"),
        (
            "sum(valid_arr[max(0, i+1-PLUSED_GIVEUP): i])",
            "int(np.sum(valid_arr[max(0, i+1-PLUSED_GIVEUP): i], dtype=np.int64))",
        ),
        ("sum(valid_arr)", "int(np.sum(valid_arr, dtype=np.int64))"),
        ("sum(arrcount)", "int(np.sum(arrcount, dtype=np.int64))"),
        (
            "func_invok[func] += test_func_arrcount[func][i]",
            "func_invok[func] += int(test_func_arrcount[func][i])",
        ),
    ]
    for old, new in int64_sums:
        if old not in src:
            raise SystemExit("int64 patch missed %s" % old)
        src = src.replace(old, new)
    if "func_candi.json" in src or "mid-data/cal_corr" in src or "mid-data/func_candi" in src:
        raise SystemExit("redirect left a mid-data correlation path in the executed source")
    return src


def _round(value, digits):
    return float(round(value, digits))


def write_summary(tag, spec, ns):
    func_cold = ns["func_cold"]
    func_invok = ns["func_invok"]
    cold_ratio = [cold / func_invok[func] for func, cold in func_cold.items()]
    n = len(cold_ratio)
    q3 = float(np.percentile(cold_ratio, 75))
    med = float(np.percentile(cold_ratio, 50))
    p90 = float(np.percentile(cold_ratio, 90))
    zero = sum(1 for func, cold in func_cold.items() if cold == 0)
    always = sum(1 for func, cold in func_cold.items() if cold == func_invok[func])
    total_cold = int(sum(int(v) for v in func_cold.values()))
    total_inv = int(sum(int(v) for v in func_invok.values()))
    arr_inv = int(sum(int(a.sum(dtype=np.int64)) for a in ns["test_func_arrcount"].values()))
    corr = ns["test_func_corr"]
    n_corr_funcs = sum(1 for v in corr.values() if len(v) > 0)
    n_corr_links = int(sum(len(v) for v in corr.values()))
    stats_path = os.path.join(ROOT, spec["data_dir"], "convert_stats.json")
    with open(stats_path) as f:
        convert_stats = json.load(f)
    summary = {
        "column": spec["column"],
        # Only the offline candidate map is empty; same-owner online links still pre-warm.
        "correlation": (
            "offline map empty; same-owner online pre-warm on"
            if n_corr_links
            else "offline map empty; no same-owner links"
        ),
        "candidate_map": "empty",
        "func_candi_json": "not read",
        "comparable_to_azure_2019_paper": False,
        "n_functions_train": int(ns["train_NUM"]),
        "n_functions_simulated": n,
        "Q3_CSR": _round(q3, 4),
        "Q3_CSR_exact": q3,
        "median_CSR": _round(med, 4),
        "median_CSR_exact": med,
        "P90_CSR": _round(p90, 4),
        "P90_CSR_exact": p90,
        "zero_cold_functions": int(zero),
        "zero_cold_share_pct": _round(100.0 * zero / n, 2),
        "always_cold_functions": int(always),
        "always_cold_share_pct": _round(100.0 * always / n, 2),
        "always_cold_definition": "cold count == invocation count (per-function CSR = 1)",
        "zero_cold_definition": "cold count == 0",
        "total_cold_starts": total_cold,
        "total_invocations": total_inv,
        "test_array_invocation_sum": arr_inv,
        "invocations_match_test_arrays": total_inv == arr_inv,
        "WMT_minutes": int(ns["waste_mem_time"]),
        "test_only_owner_app_corr_functions": n_corr_funcs,
        "test_only_owner_app_corr_links": n_corr_links,
        "perform_pruning_disabled": bool(ns.get("perform_pruning_disabled", False)),
        "convert": convert_stats,
    }
    if ns.get("inflation"):
        summary["inflation"] = ns["inflation"]
    path = os.path.join(ROOT, spec["summary"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=4, sort_keys=True)
        f.write("\n")
    print(
        tag,
        "simulated", n,
        "Q3", summary["Q3_CSR"],
        "zero%", summary["zero_cold_share_pct"],
        "always%", summary["always_cold_share_pct"],
        "WMT", summary["WMT_minutes"],
        flush=True,
    )
    return summary


def replication_shape(n_test, target):
    """Smallest K with K*n_test >= target, and the last-replica prefix that hits target."""
    if n_test <= 0:
        raise SystemExit("no test-active functions to replicate")
    copies = (target + n_test - 1) // n_test
    prefix = target - (copies - 1) * n_test
    if not 1 <= prefix <= n_test:
        raise SystemExit("prefix %s outside 1..%s" % (prefix, n_test))
    if (copies - 1) * n_test + prefix != target:
        raise SystemExit("replication does not hit %s" % target)
    return copies, prefix


def replicate_test_functions(train_arr, test_arr, train_oa, test_oa, train_owner, train_app, test_owner, test_app, trig, target):
    """Repeat real test-active series until the simulated count is `target`.

    Copy 1 is the original funcID. Copies 2..K-1 are full replicas suffixed
    #r2, #r3, .... Copy K is the lexicographic prefix of length P. Count
    buffers are shared and only read by the simulator.
    """
    ids = sorted(test_arr)
    copies, prefix = replication_shape(len(ids), target)
    if copies == 1:
        for fid in ids[prefix:]:
            test_arr.pop(fid)
        kept = ids[:prefix]
        return {
            "S_test_active": len(ids),
            "K": copies,
            "full_copies_including_original": 0,
            "last_replica_suffix": None,
            "last_replica_prefix": prefix,
            "prefix_order": "funcID lexicographic",
            "prefix_first": kept[0],
            "prefix_last": kept[-1],
            "simulated_functions": prefix,
            "arrays": "originals only; dropped the lexicographic tail",
        }
    plan = [(r, ids) for r in range(2, copies)]
    plan.append((copies, ids[:prefix]))
    n_test_only = 0
    for fid in ids:
        if fid not in train_arr:
            n_test_only += 1
    for replica, subset in plan:
        suffix = "#r%d" % replica
        for fid in subset:
            new = fid + suffix
            if new in test_arr or new in train_arr:
                raise SystemExit("replica id collides with a real funcID: %s" % new)
            test_arr[new] = test_arr[fid]
            test_oa[new] = test_oa[fid]
            owner, app = test_oa[fid].split("\t")
            test_owner[owner].add(new)
            test_app[app].add(new)
            trig[new] = set(trig.get(fid, ()))
            if fid in train_arr:
                train_arr[new] = train_arr[fid]
                train_oa[new] = train_oa[fid]
                train_owner[owner].add(new)
                train_app[app].add(new)
    if len(test_arr) != target:
        raise SystemExit("inflated test functions %s != %s" % (len(test_arr), target))
    return {
        "method": "replication of real series",
        "S_test_active": len(ids),
        "K": copies,
        "full_copies_including_original": copies - 1,
        "replica_suffixes": ["#r%d" % r for r in range(2, copies + 1)],
        "last_replica_suffix": "#r%d" % copies,
        "last_replica_prefix": prefix,
        "prefix_order": "funcID lexicographic",
        "prefix_first": ids[0],
        "prefix_last": ids[prefix - 1],
        "simulated_functions": len(test_arr),
        "test_only_originals": n_test_only,
        "owner": "same funcID string as the original",
        "trigger": "same as the original",
        "arrays": "shared read-only int32 buffers; each replica is its own function key",
    }


def run(tag):
    spec = DATASETS[tag]
    data_dir = os.path.join(ROOT, spec["data_dir"])
    prefix = os.path.join(data_dir, "invocations_per_function_md.anon.d")
    train_files = [prefix + "%02d.csv" % i for i in range(1, 13)]
    test_files = [prefix + "13.csv", prefix + "14.csv"]
    for path in train_files + test_files:
        if not os.path.isfile(path):
            raise SystemExit("missing %s" % path)
    print("load", tag, flush=True)
    train_func_arrcount, train_func_owner_app, train_owner_func, train_app_func, func_trigger = load_split(train_files, 12)
    test_func_arrcount, test_func_owner_app, test_owner_func, test_app_func, test_trig = load_split(test_files, 2)
    for func, trigs in test_trig.items():
        func_trigger[func] |= trigs
    inflation = None
    if spec.get("inflate_to"):
        assert replication_shape(10000, 39388) == (4, 9388)
        assert replication_shape(13247, 39388) == (3, 12894)
        inflation = replicate_test_functions(
            train_func_arrcount,
            test_func_arrcount,
            train_func_owner_app,
            test_func_owner_app,
            train_owner_func,
            train_app_func,
            test_owner_func,
            test_app_func,
            func_trigger,
            spec["inflate_to"],
        )
        print("inflation", json.dumps(inflation), flush=True)
    random.seed(0)
    np.random.seed(0)
    ns = {
        "os": os,
        "sys": sys,
        "json": json,
        "time": __import__("time"),
        "random": random,
        "bisect": __import__("bisect"),
        "defaultdict": defaultdict,
        "np": np,
        "pd": pd,
        "plt": plt,
        "tqdm": tqdm,
        "train_func_arrcount": train_func_arrcount,
        "train_func_owner_app": train_func_owner_app,
        "train_owner_func": train_owner_func,
        "train_app_func": train_app_func,
        "func_trigger": func_trigger,
        "test_func_arrcount": test_func_arrcount,
        "test_func_owner_app": test_func_owner_app,
        "test_owner_func": test_owner_func,
        "test_app_func": test_app_func,
        "train_NUM": len(train_func_arrcount),
        "test_NUM": len(test_func_arrcount),
        "inflation": inflation,
    }
    print("loaded", ns["train_NUM"], ns["test_NUM"], "exec", tag, flush=True)
    src = algorithm_source(tag)
    os.chdir(ROOT)
    run_dir = os.path.join(ROOT, "result", "runs", tag)
    os.makedirs(run_dir, exist_ok=True)
    executed = os.path.join(run_dir, "executed.py")
    with open(executed, "w", encoding="utf-8") as f:
        f.write(src)
    exec(compile(src, executed, "exec"), ns)
    return write_summary(tag, spec, ns)


def main():
    os.chdir(ROOT)
    args = sys.argv[1:] or ["all"]
    if args == ["all"]:
        tags = ["azure2021", "huawei"]
    else:
        tags = args
    failed = []
    for tag in tags:
        if tag not in DATASETS:
            raise SystemExit("unknown dataset %s" % tag)
        try:
            run(tag)
        except Exception:
            traceback.print_exc()
            failed.append(tag)
    if failed:
        raise SystemExit("failed: %s" % ", ".join(failed))


if __name__ == "__main__":
    main()
