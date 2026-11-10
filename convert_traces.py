#!/usr/bin/env python3
"""Write SPES daily invocation CSVs for Azure 2021 and Huawei 2025 Region 5.

SPES columns: HashOwner, HashApp, HashFunction, Trigger, then 1440 minute counts.
Counts are int32. Days are d01.csv–d14.csv (12 train + 2 test).

Azure 2021: one row per invocation. Minute is floor(end_timestamp/60), clipped
into the 14*1440 grid. Function key is app+func (fixed-width hex). Owner and
application are the app id. Trigger is "unknown".

Huawei: Region 5 quantile `requests` only (per-minute invocation counts).
Days 0–11 → d01–d12, days 12–13 → d13–d14 (of a 31-day series). Each funcID
is its own owner and application. Trigger is "unknown".
"""

import csv
import io
import json
import math
import os
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
N_DAYS = 14
MINUTES = N_DAYS * 1440
INT32_MAX = int(np.iinfo(np.int32).max)

AZ_TXT = os.path.join(
    ROOT,
    "raw_data/azurefunctions-invocation-2021/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt",
)
AZ_OUT = os.path.join(ROOT, "raw_data/azurefunctions-2021-spes")
HW_ZIP = os.path.join(ROOT, "raw_data/huawei-2025/region5_quantiles.zip")
HW_OUT = os.path.join(ROOT, "raw_data/huawei2025-spes")


def clip_minute(timestamp, n_minutes=MINUTES):
    """floor(timestamp/60), clipped into [0, n_minutes). Returns (minute, clipped)."""
    minute = math.floor(timestamp / 60.0)
    if minute < 0:
        return 0, True
    if minute >= n_minutes:
        return n_minutes - 1, True
    return minute, False


def _header():
    return ["HashOwner", "HashApp", "HashFunction", "Trigger"] + [str(i) for i in range(1, 1441)]


def _store(grid, minute, value, max_cell, n_clip):
    if value > INT32_MAX:
        n_clip += 1
        value = INT32_MAX
    if value > max_cell:
        max_cell = value
    grid[minute] = value
    return max_cell, n_clip


def write_days(grids, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    header = _header()
    keys = sorted(grids)
    for day in range(N_DAYS):
        path = os.path.join(out_dir, "invocations_per_function_md.anon.d%02d.csv" % (day + 1))
        sl = slice(day * 1440, (day + 1) * 1440)
        n = 0
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            for key in keys:
                counts = grids[key][sl]
                if not counts.any():
                    continue
                owner, app, func, trig = meta[key]
                w.writerow([owner, app, func, trig, *counts.tolist()])
                n += 1
        print(path, "rows", n, flush=True)


def check_written(out_dir, expected_total):
    total = 0
    for day in range(1, N_DAYS + 1):
        path = os.path.join(out_dir, "invocations_per_function_md.anon.d%02d.csv" % day)
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header[:4] != ["HashOwner", "HashApp", "HashFunction", "Trigger"] or len(header) != 1444:
                raise SystemExit("bad header %s" % path)
            seen = set()
            for row in reader:
                if len(row) != 1444:
                    raise SystemExit("bad width %s" % path)
                if row[2] in seen:
                    raise SystemExit("duplicate function %s in %s" % (row[2], path))
                seen.add(row[2])
                total += sum(int(x) for x in row[4:])
    if total != expected_total:
        raise SystemExit("sum %s != expected %s in %s" % (total, expected_total, out_dir))
    return total


def convert_azure():
    grids = {}
    meta = {}
    max_cell = 0
    n_clip = 0
    n_oob = 0
    n_events = 0
    with open(AZ_TXT, newline="") as f:
        for row in csv.DictReader(f):
            n_events += 1
            app, func = row["app"], row["func"]
            if len(app) != 64 or len(func) != 64:
                raise SystemExit("expected 64-hex app and func, got %s %s" % (len(app), len(func)))
            minute, clipped = clip_minute(float(row["end_timestamp"]))
            n_oob += int(clipped)
            key = app + func
            grid = grids.get(key)
            if grid is None:
                grid = np.zeros(MINUTES, dtype=np.int32)
                grids[key] = grid
                meta[key] = (app, app, key, "unknown")
            value = int(grid[minute]) + 1
            max_cell, n_clip = _store(grid, minute, value, max_cell, n_clip)
            if n_events % 500000 == 0:
                print("azure rows", n_events, "funcs", len(grids), flush=True)
    write_days(grids, meta, AZ_OUT)
    # Each event adds 1. A clip saturates the cell, so the written sum drops.
    written = check_written(AZ_OUT, n_events - n_clip)
    stats = {
        "dataset": "azure-2021",
        "events": n_events,
        "functions": len(grids),
        "max_minute_count": max_cell,
        "int32_clipped_increments": n_clip,
        "timestamps_clipped_into_grid": n_oob,
        "written_invocation_sum": written,
        "function_key": "app+func concatenated 64-hex",
        "owner": "app",
        "application": "app",
        "trigger": "unknown",
        "train_days": "d01-d12",
        "test_days": "d13-d14",
        "minute": "floor(end_timestamp/60) clipped into 14*1440",
    }
    _dump_stats(AZ_OUT, stats)
    print("azure max_minute_count", max_cell, "clip", n_clip, "oob", n_oob, flush=True)
    return stats


def convert_huawei():
    grids = {}
    meta = {}
    max_cell = 0
    n_clip = 0
    n_nonint = 0
    expected = 0
    with zipfile.ZipFile(HW_ZIP) as zf:
        for day in range(N_DAYS):
            name = "R5/requests/day_%02d.csv" % day
            with zf.open(name) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, newline="", encoding="utf-8"))
                header = next(reader)
                func_ids = header[2:]
                columns = {}
                for j, fid in enumerate(func_ids):
                    if "\t" in fid or "\n" in fid or "\r" in fid:
                        raise SystemExit("funcID has whitespace: %r" % fid)
                    columns.setdefault(fid, []).append(j)
                    if fid not in grids:
                        grids[fid] = np.zeros(MINUTES, dtype=np.int32)
                        meta[fid] = (fid, fid, fid, "unknown")
                seen = set()
                for row in reader:
                    if len(row) < len(header):
                        row = row + [""] * (len(header) - len(row))
                    if int(float(row[0])) != day:
                        raise SystemExit("%s day column %s" % (name, row[0]))
                    sec = int(float(row[1]))
                    if sec % 60 != 0:
                        raise SystemExit("%s time %s is not a minute" % (name, row[1]))
                    # time is seconds from the start of the trace, not within the day.
                    minute = sec // 60 - day * 1440
                    if minute in seen or minute < 0 or minute >= 1440:
                        raise SystemExit("%s bad minute %s (time %s)" % (name, minute, sec))
                    seen.add(minute)
                    dest = day * 1440 + minute
                    for fid, idxs in columns.items():
                        acc = 0
                        for j in idxs:
                            cell = row[2 + j]
                            if cell == "":
                                continue
                            value = float(cell)
                            if abs(value - round(value)) > 1e-6:
                                n_nonint += 1
                            iv = int(round(value))
                            if iv < 0:
                                iv = 0
                            acc += iv
                        if acc == 0:
                            continue
                        max_cell, n_clip = _store(grids[fid], dest, acc, max_cell, n_clip)
                        expected += int(grids[fid][dest])
                if seen != set(range(1440)):
                    raise SystemExit("%s missing %s minutes" % (name, 1440 - len(seen)))
            print("huawei", name, "funcs_so_far", len(grids), flush=True)
    write_days(grids, meta, HW_OUT)
    written = check_written(HW_OUT, expected)
    stats = {
        "dataset": "huawei-2025-r5",
        "region": "R5",
        "metric": "requests",
        "source_days": "day_00-day_13 of day_00-day_30",
        "functions": len(grids),
        "max_minute_count": max_cell,
        "int32_clipped_cells": n_clip,
        "non_integer_cells_rounded": n_nonint,
        "written_invocation_sum": written,
        "function_key": "funcID",
        "owner": "funcID (one function per owner)",
        "application": "funcID (one function per application)",
        "trigger": "unknown",
        "train_days": "day_00-day_11 as d01-d12",
        "test_days": "day_12-day_13 as d13-d14",
    }
    _dump_stats(HW_OUT, stats)
    print("huawei max_minute_count", max_cell, "clip", n_clip, "nonint", n_nonint, flush=True)
    return stats


def _dump_stats(out_dir, stats):
    with open(os.path.join(out_dir, "convert_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")


def _self_check_clip():
    assert clip_minute(0.07949090003967285) == (0, False)
    assert clip_minute(60) == (1, False)
    assert clip_minute(59.9) == (0, False)
    assert clip_minute(-0.1) == (0, True)
    assert clip_minute(MINUTES * 60) == (MINUTES - 1, True)
    assert clip_minute(1209599.6944639683) == (20159, False)


def main():
    os.chdir(ROOT)
    _self_check_clip()
    convert_azure()
    convert_huawei()


if __name__ == "__main__":
    main()
