"""Run cal_corr shards one after another, then read, re-read, compare, and graph.

The correlation rule is cal_corr.cal_correlation / cal_corr.run_shard. This script
only changes scheduling: one process, one shard file at a time, then a check that
a second full read matches the first.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cal_corr import load_corr_inputs, run_shard


def shard_count(n_funcs, total):
    if n_funcs <= 0:
        return 0
    return (n_funcs + total - 1) // total


def parse_shard(path):
    rows = {}
    with open(path) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            func, tri_tuple_lst = line.split(":", 1)
            tuples = []
            if tri_tuple_lst:
                for tri_tuple_str in tri_tuple_lst.split("\t"):
                    candi, lag, corr_rate = tri_tuple_str.split(",")
                    tuples.append((candi, int(lag), float(corr_rate)))
            rows[func] = tuples
    return rows


def read_each(paths):
    per_file = {}
    for path in paths:
        per_file[os.path.basename(path)] = parse_shard(path)
    return per_file


def read_all(paths):
    merged = {}
    for path in paths:
        for func, tuples in parse_shard(path).items():
            if func in merged:
                raise SystemExit(f"function {func} appears in more than one shard")
            merged[func] = tuples
    return merged


def compare(per_file, merged):
    again = {}
    stats = []
    mismatches = []
    for name, rows in per_file.items():
        n_edges = sum(len(v) for v in rows.values())
        stats.append({"file": name, "functions": len(rows), "edges": n_edges})
        for func, tuples in rows.items():
            again[func] = tuples
            if merged.get(func) != tuples:
                mismatches.append(func)
    only_again = sorted(set(again) - set(merged))
    only_merged = sorted(set(merged) - set(again))
    return {
        "ok": not mismatches and not only_again and not only_merged and again == merged,
        "n_functions": len(merged),
        "n_edges": sum(len(v) for v in merged.values()),
        "mismatch_functions": mismatches,
        "only_in_reread": only_again,
        "only_in_first_merge": only_merged,
        "per_file": stats,
    }


def graph(stats, out_png):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    labels = [s["file"].replace(".txt", "") for s in stats]
    funcs = [s["functions"] for s in stats]
    edges = [s["edges"] for s in stats]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, funcs, color="#2A4597")
    axes[0].set_title("Functions with a candidate (per shard)")
    axes[0].set_xlabel("shard")
    axes[0].set_xticks(list(x), labels, rotation=90, fontsize=8)
    axes[1].bar(x, edges, color="#C5563B")
    axes[1].set_title("Candidate edges (per shard)")
    axes[1].set_xlabel("shard")
    axes[1].set_xticks(list(x), labels, rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_merged_json(merged, path):
    # Same tuple fields as merge_corr_res.py, written to a separate file.
    obj = {func: [[c, lag, rate] for (c, lag, rate) in tuples] for func, tuples in sorted(merged.items())}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fw:
        json.dump(obj, fw, indent=4, separators=(",", ": "), ensure_ascii=False)


def self_check(tmp):
    corr = os.path.join(tmp, "cal_corr")
    out = os.path.join(tmp, "shards")
    os.makedirs(corr, exist_ok=True)
    # forget=0 so the slice is the whole series. Pattern: B tracks A at lag 0.
    series = {
        "A": [1, 0, 1, 0, 1, 0, 1, 0],
        "B": [1, 0, 1, 0, 1, 0, 1, 0],
        "C": [0, 1, 0, 1, 0, 1, 0, 1],
    }
    with open(os.path.join(corr, "to_cal_corr.txt"), "w") as wf:
        for func, arr in series.items():
            wf.write(func + "\t0\t" + ",".join(str(x) for x in arr) + "\n")
    with open(os.path.join(corr, "unknown_ownerapp_lst.txt"), "w") as wf:
        wf.write("A\tB,C\n")
        wf.write("C\tA\n")
    return corr, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corr-dir", default="mid-data/cal_corr")
    parser.add_argument("--out-dir", default="mid-data/func_candi_tuple_lst_seq")
    parser.add_argument("--total", default=3704, type=int)
    parser.add_argument("--warm-th", default=0.7, type=float)
    parser.add_argument("--report", default="result/corr_seq_compare.json")
    parser.add_argument("--graph", default="result/corr_seq_compare.png")
    parser.add_argument("--merged-json", default="mid-data/func_candi_seq.json")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="corr_seq_")
        args.corr_dir, args.out_dir = self_check(tmp)
        args.total = 1
        args.report = os.path.join(tmp, "compare.json")
        args.graph = os.path.join(tmp, "compare.png")
        args.merged_json = os.path.join(tmp, "func_candi_seq.json")
        print("self-check dir", tmp)

    func_arrcount, func_forget, unknown = load_corr_inputs(args.corr_dir)
    n_shards = shard_count(len(unknown), args.total)
    print(f"functions {len(unknown)} shards {n_shards} total {args.total}", flush=True)

    paths = []
    for start in range(n_shards):
        print(f"shard {start}/{n_shards}", flush=True)
        paths.append(run_shard(start, args.total, args.warm_th, func_arrcount, func_forget, unknown, args.out_dir))

    first = read_each(paths)
    second_merged = read_all(paths)
    # Re-read every file again and compare to the first parse and to the merged view.
    reread = read_each(paths)
    report = compare(reread, second_merged)
    report["first_read_matches_reread"] = first == reread
    report["ok"] = report["ok"] and report["first_read_matches_reread"]
    graph(report["per_file"], args.graph)
    write_merged_json(second_merged, args.merged_json)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fw:
        json.dump(report, fw, indent=2)
    print(json.dumps({k: report[k] for k in ("ok", "n_functions", "n_edges", "first_read_matches_reread")}, indent=2))
    print("graph", args.graph)
    print("merged", args.merged_json)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
