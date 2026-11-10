#!/usr/bin/env python3
"""Write SPES daily CSVs for the lexicographic first half of Globus Compute.

Half: function_uuid values from functions.csv, sorted lexicographically, first
n//2. Tasks whose function_uuid is outside that set are dropped.

Minute counts use the task `received` time. When `received` is empty, the
earliest of the other timestamp columns is used. Values above 1e16 are
nanoseconds since the Unix epoch (the file's actual unit).

SPES split: the earliest 14 UTC calendar days that contain at least one kept
task. Days 1-12 are train, days 13-14 are test. Owner and application are the
task's anonymized_user_uuid, or the function_uuid when that field is empty.
Trigger is "unknown". Counts are int32.
"""

import csv
import json
import os
from collections import Counter

import numpy as np

from convert_traces import INT32_MAX, MINUTES, N_DAYS, check_written, write_days

ROOT = os.path.dirname(os.path.abspath(__file__))
FUNC_CSV = os.path.join(ROOT, "raw_data/globus-compute/GlobusComputeDataset-v1-functions.csv")
TASK_CSV = os.path.join(ROOT, "raw_data/globus-compute/GlobusComputeDataset-v1-tasks.csv")
OUT = os.path.join(ROOT, "raw_data/globus-half-spes")

# received is preferred. The rest are fallbacks, earliest non-empty value.
TS_COLS = (
    "received",
    "waiting_for_nodes",
    "waiting_for_launch",
    "execution_start",
    "execution_end",
    "result_received",
)


def epoch_seconds(text):
    """Parse a Globus timestamp cell to Unix seconds, or None if empty."""
    if text is None:
        return None
    text = text.strip()
    if text == "":
        return None
    raw = float(text) if ("." in text or "e" in text or "E" in text) else int(text)
    if raw <= 0:
        return None
    if raw > 1e16:
        return raw / 1e9
    if raw > 1e14:
        return raw / 1e6
    if raw > 1e11:
        return raw / 1e3
    return float(raw)


def half_function_ids(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "function_uuid" not in reader.fieldnames:
            raise SystemExit("functions.csv missing function_uuid: %s" % (reader.fieldnames,))
        uuids = [row["function_uuid"] for row in reader]
    if not uuids:
        raise SystemExit("functions.csv is empty")
    n_unique = len(set(uuids))
    uuids.sort()
    half_n = len(uuids) // 2
    chosen = uuids[:half_n]
    # A duplicated uuid sitting on the cut would also appear in the discarded half.
    overlap = set(chosen) & set(uuids[half_n:])
    if overlap:
        raise SystemExit("function_uuid duplicated across the half cut: %s" % len(overlap))
    return set(chosen), len(uuids), n_unique, half_n, chosen[0], chosen[-1]


def task_seconds(row, idx):
    """Return (seconds, used_fallback). received wins; else the earliest other stamp."""
    received = epoch_seconds(row[idx["received"]])
    if received is not None:
        return received, False
    best = None
    for col in TS_COLS[1:]:
        sec = epoch_seconds(row[idx[col]])
        if sec is not None and (best is None or sec < best):
            best = sec
    return best, True


def _indexes(header):
    missing = [c for c in ("function_uuid", "anonymized_user_uuid") + TS_COLS if c not in header]
    if missing:
        raise SystemExit("tasks.csv missing columns %s; header=%s" % (missing, header))
    return {name: header.index(name) for name in header}


def scan_days(half, idx):
    """First pass: UTC day numbers (sec // 86400) that contain a kept task."""
    days = Counter()
    n_rows = 0
    n_kept = 0
    n_fallback = 0
    n_no_ts = 0
    with open(TASK_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if [header[idx[c]] for c in idx] != list(idx):
            # idx was built from this same header order; just guard a reread.
            pass
        file_idx = _indexes(header)
        for row in reader:
            n_rows += 1
            func = row[file_idx["function_uuid"]]
            if func not in half:
                continue
            sec, fallback = task_seconds(row, file_idx)
            if sec is None:
                n_no_ts += 1
                continue
            n_kept += 1
            n_fallback += int(fallback)
            days[int(sec // 86400)] += 1
            if n_rows % 500000 == 0:
                print("globus pass1 rows", n_rows, "kept", n_kept, flush=True)
    if not days:
        raise SystemExit("no kept tasks with a timestamp")
    return days, n_rows, n_kept, n_fallback, n_no_ts


def bucket(half, window):
    """Second pass: per-minute int32 counts inside the 14-day window."""
    day_index = {day: i for i, day in enumerate(window)}
    grids = {}
    users = {}
    max_raw = 0
    n_clip = 0
    n_in_window = 0
    n_rows = 0
    with open(TASK_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        file_idx = _indexes(header)
        for row in reader:
            n_rows += 1
            func = row[file_idx["function_uuid"]]
            if func not in half:
                continue
            sec, _fallback = task_seconds(row, file_idx)
            if sec is None:
                continue
            day = int(sec // 86400)
            slot = day_index.get(day)
            if slot is None:
                continue
            n_in_window += 1
            user = row[file_idx["anonymized_user_uuid"]].strip() or func
            grid = grids.get(func)
            if grid is None:
                grid = np.zeros(MINUTES, dtype=np.int32)
                grids[func] = grid
                users[func] = Counter()
            users[func][user] += 1
            minute = slot * 1440 + int((sec % 86400) // 60)
            value = int(grid[minute]) + 1
            if value > max_raw:
                max_raw = value
            if value > INT32_MAX:
                n_clip += 1
                value = INT32_MAX
            grid[minute] = np.int32(value)
            if n_rows % 500000 == 0:
                print("globus pass2 rows", n_rows, "window_tasks", n_in_window, "funcs", len(grids), flush=True)
    meta = {}
    n_multi = 0
    for func, counts in users.items():
        # Most common user. Ties: lexicographically smallest uuid.
        winner = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        if len(counts) > 1:
            n_multi += 1
        meta[func] = (winner, winner, func, "unknown")
    return grids, meta, n_in_window, max_raw, n_clip, n_multi


def _iso(day_number):
    import datetime
    return datetime.datetime.fromtimestamp(day_number * 86400, datetime.timezone.utc).date().isoformat()


def main():
    os.chdir(ROOT)
    half, n_rows_func, n_unique, half_n, first_id, last_id = half_function_ids(FUNC_CSV)
    print(
        "functions", n_rows_func, "unique", n_unique, "half", half_n,
        "first", first_id, "last_in_half", last_id, flush=True,
    )
    # Header once, then the scan uses its own reader.
    with open(TASK_CSV, newline="") as f:
        header = next(csv.reader(f))
    idx = _indexes(header)
    days, n_task_rows, n_kept, n_fallback, n_no_ts = scan_days(half, idx)
    ordered = sorted(days)
    window = ordered[:N_DAYS]
    if len(window) < N_DAYS:
        raise SystemExit("only %s UTC days contain kept tasks; need %s" % (len(window), N_DAYS))
    contiguous = all(window[i] + 1 == window[i + 1] for i in range(len(window) - 1))
    print(
        "kept_tasks", n_kept, "fallback", n_fallback, "no_ts", n_no_ts,
        "active_days", len(ordered), "window", _iso(window[0]), _iso(window[-1]),
        "contiguous", contiguous, flush=True,
    )
    grids, meta, n_in_window, max_raw, n_clip, n_multi = bucket(half, window)
    test_sum = 0
    for grid in grids.values():
        test_sum += int(grid[12 * 1440:].sum(dtype=np.int64))
    expected = int(sum(int(g.sum(dtype=np.int64)) for g in grids.values()))
    write_days(grids, meta, OUT)
    written = check_written(OUT, expected)
    stats = {
        "dataset": "globus-compute-half",
        "half": "first half of function_uuid values sorted lexicographically (n//2)",
        "functions_in_functions_csv": n_rows_func,
        "unique_function_uuid_in_functions_csv": n_unique,
        "functions_in_half": len(half),
        "half_cut_rows": half_n,
        "half_first_function_uuid": first_id,
        "half_last_function_uuid": last_id,
        "task_rows": n_task_rows,
        "tasks_kept_half": n_kept,
        "tasks_received_missing_used_earliest_other_timestamp": n_fallback,
        "tasks_kept_with_no_timestamp": n_no_ts,
        "tasks_in_14day_window": n_in_window,
        "active_utc_days_in_half": len(ordered),
        "window_utc_dates": [_iso(d) for d in window],
        "window_contiguous_utc_days": contiguous,
        "train_days": "earliest 12 UTC days with kept tasks as d01-d12",
        "test_days": "next 2 UTC days with kept tasks as d13-d14",
        "functions": len(grids),
        "functions_with_multiple_users_in_window": n_multi,
        "max_minute_count": int(max_raw),
        "int32_clipped_increments": n_clip,
        "written_invocation_sum": written,
        "test_day_invocation_sum": test_sum,
        "function_key": "function_uuid",
        "owner": "anonymized_user_uuid, or function_uuid when the task user is empty",
        "application": "anonymized_user_uuid, or function_uuid when the task user is empty",
        "trigger": "unknown",
        "minute": "UTC minute of received (else earliest other timestamp); days are UTC dates",
        "timestamp_column": "received",
        "correlation": "disabled",
        "func_candi_json": "not read",
    }
    with open(os.path.join(OUT, "convert_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        "globus functions_in_window", len(grids),
        "invocations", written,
        "test_invocations", test_sum,
        "max_minute", max_raw,
        "int32_clipped", n_clip,
        flush=True,
    )
    if n_clip:
        print("CLIPPED minute counts to int32; unclipped max was", max_raw, flush=True)
    return stats


if __name__ == "__main__":
    main()
