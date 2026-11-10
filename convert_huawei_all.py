#!/usr/bin/env python3
"""Union Huawei 2025 regions R1–R5 into SPES daily CSVs.

Streams each region's quantile `requests` days 0–13 from the zip (one region
at a time). Other zip members are not extracted. Counts stay int32. A funcID
that appears in more than one region is one function; minute counts are summed.
"""

import csv
import io
import os
import zipfile

import numpy as np

from convert_traces import (
    INT32_MAX,
    MINUTES,
    N_DAYS,
    _dump_stats,
    check_written,
    write_days,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
HW_DIR = os.path.join(ROOT, "raw_data/huawei-2025")
OUT = os.path.join(ROOT, "raw_data/huawei2025-all-spes")
# R5 first so the known single-region totals can be checked before the union grows.
REGIONS = (5, 3, 2, 1, 4)
R5_FUNCTIONS = 543
R5_MAX = 18119
R5_SUM = 56401079


def _rss_mb():
    import resource

    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _parse_count(cell, n_nonint):
    value = float(cell)
    if value != value or value in (float("inf"), float("-inf")):
        raise SystemExit("non-finite requests cell %r" % cell)
    if abs(value - round(value)) > 1e-6:
        n_nonint += 1
    iv = int(round(value))
    if iv < 0:
        iv = 0
    return iv, n_nonint


def _add(grid, dest, iv, n_clip, max_cell):
    if iv <= 0:
        return n_clip, max_cell
    total = int(grid[dest]) + iv
    if total > INT32_MAX:
        n_clip += 1
        total = INT32_MAX
    grid[dest] = total
    if total > max_cell:
        max_cell = total
    return n_clip, max_cell


def _ingest_region(region, grids, meta, regions_of):
    path = os.path.join(HW_DIR, "region%d_quantiles.zip" % region)
    if not os.path.isfile(path):
        raise SystemExit("missing %s" % path)
    n_clip = 0
    n_nonint = 0
    max_cell = 0
    added = 0
    column_counts = []
    unique = set()
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for day in range(N_DAYS):
            name = "R%d/requests/day_%02d.csv" % (region, day)
            if name not in names:
                raise SystemExit("missing %s in %s" % (name, path))
            with zf.open(name) as raw:
                text = io.TextIOWrapper(raw, newline="", encoding="utf-8")
                rows = csv.reader(text)
                header = next(rows)
                if header[:2] != ["day", "time"]:
                    raise SystemExit("%s header %s" % (name, header[:2]))
                func_ids = header[2:]
                column_counts.append(len(func_ids))
                cols = []
                for fid in func_ids:
                    if "\t" in fid or "\n" in fid or "\r" in fid or "#" in fid:
                        raise SystemExit("funcID not usable as a SPES key: %r" % fid)
                    unique.add(fid)
                    grid = grids.get(fid)
                    if grid is None:
                        grid = np.zeros(MINUTES, dtype=np.int32)
                        grids[fid] = grid
                        meta[fid] = (fid, fid, fid, "unknown")
                    seen = regions_of.get(fid)
                    if seen is None:
                        regions_of[fid] = {region}
                    else:
                        seen.add(region)
                    cols.append(grid)
                seen_min = set()
                for row in rows:
                    if len(row) < len(header):
                        row = row + [""] * (len(header) - len(row))
                    if int(float(row[0])) != day:
                        raise SystemExit("%s day column %s" % (name, row[0]))
                    sec = int(float(row[1]))
                    if sec % 60 != 0:
                        raise SystemExit("%s time %s is not a minute" % (name, row[1]))
                    minute = sec // 60 - day * 1440
                    if minute in seen_min or minute < 0 or minute >= 1440:
                        raise SystemExit("%s bad minute %s (time %s)" % (name, minute, sec))
                    seen_min.add(minute)
                    dest = day * 1440 + minute
                    for j, cell in enumerate(row[2:2 + len(cols)]):
                        if cell == "":
                            continue
                        iv, n_nonint = _parse_count(cell, n_nonint)
                        before = int(cols[j][dest])
                        n_clip, max_cell = _add(cols[j], dest, iv, n_clip, max_cell)
                        added += int(cols[j][dest]) - before
                if seen_min != set(range(1440)):
                    raise SystemExit("%s missing %s minutes" % (name, 1440 - len(seen_min)))
                text.detach()
            print(
                "region", region, name, "cols", len(func_ids),
                "union", len(grids), "rss_mb", round(_rss_mb(), 1),
                flush=True,
            )
    return {
        "region": "R%d" % region,
        "column_counts_day_00_to_13": column_counts,
        "column_count": column_counts[0] if len(set(column_counts)) == 1 else None,
        "unique_funcIDs": len(unique),
        "invocation_sum_added": int(added),
        "max_minute_after_region": int(max_cell),
        "int32_clipped_cells_so_far": int(n_clip),
        "non_integer_cells_rounded": int(n_nonint),
    }


def _snapshot(grids):
    total = 0
    test = 0
    window = 0
    test_funcs = 0
    max_cell = 0
    for grid in grids.values():
        s = int(grid.sum(dtype=np.int64))
        total += s
        if s:
            window += 1
        t = int(grid[12 * 1440:].sum(dtype=np.int64))
        test += t
        if t:
            test_funcs += 1
        m = int(grid.max()) if s else 0
        if m > max_cell:
            max_cell = m
    return {
        "lifetime_unique_funcIDs": len(grids),
        "functions_in_14day_window": window,
        "functions_with_test_day_counts": test_funcs,
        "written_invocation_sum": total,
        "test_day_invocations": test,
        "max_minute_count": max_cell,
    }


def convert():
    grids = {}
    meta = {}
    regions_of = {}
    per_region = []
    n_clip = 0
    n_nonint = 0
    for region in REGIONS:
        info = _ingest_region(region, grids, meta, regions_of)
        n_clip += info["int32_clipped_cells_so_far"]
        n_nonint += info["non_integer_cells_rounded"]
        per_region.append(info)
        if region == 5:
            snap = _snapshot(grids)
            if (
                snap["lifetime_unique_funcIDs"] != R5_FUNCTIONS
                or snap["written_invocation_sum"] != R5_SUM
                or snap["max_minute_count"] != R5_MAX
                or info["int32_clipped_cells_so_far"] != 0
            ):
                raise SystemExit("R5 sanity check failed: %s clip %s" % (snap, info))
            print("R5 sanity check ok", snap["written_invocation_sum"], flush=True)
    snap = _snapshot(grids)
    overlaps = sorted(fid for fid, regs in regions_of.items() if len(regs) > 1)
    overlap_regions = {fid: sorted(regions_of[fid]) for fid in overlaps}
    expected = snap["written_invocation_sum"]
    write_days(grids, meta, OUT)
    del grids, meta
    written = check_written(OUT, expected)
    stats = {
        "dataset": "huawei-2025-all",
        "regions": ["R%d" % r for r in (1, 2, 3, 4, 5)],
        "union_complete": True,
        "regions_failed": [],
        "metric": "requests",
        "source_days": "day_00-day_13 of day_00-day_30",
        "extract": "streamed requests/day_00-day_13 from each quantile zip, one region at a time; other members not extracted",
        "lifetime_unique_funcIDs": snap["lifetime_unique_funcIDs"],
        "functions_in_14day_window": snap["functions_in_14day_window"],
        "functions_with_test_day_counts": snap["functions_with_test_day_counts"],
        "functions": snap["functions_in_14day_window"],
        "written_invocation_sum": written,
        "test_day_invocations": snap["test_day_invocations"],
        "max_minute_count": snap["max_minute_count"],
        "int32_clipped_cells": n_clip,
        "non_integer_cells_rounded": n_nonint,
        "per_region": per_region,
        "overlap_funcIDs": overlaps,
        "overlap_regions": overlap_regions,
        "overlap_policy": "one row per funcID; per-minute counts summed across regions",
        "function_key": "funcID",
        "owner": "funcID (one function per owner)",
        "application": "funcID (one function per application)",
        "trigger": "unknown",
        "train_days": "day_00-day_11 as d01-d12",
        "test_days": "day_12-day_13 as d13-d14",
    }
    _dump_stats(OUT, stats)
    print(
        "union",
        "lifetime", stats["lifetime_unique_funcIDs"],
        "window", stats["functions_in_14day_window"],
        "test_funcs", stats["functions_with_test_day_counts"],
        "invocations", stats["written_invocation_sum"],
        "test_invocations", stats["test_day_invocations"],
        "max", stats["max_minute_count"],
        "clip", stats["int32_clipped_cells"],
        "overlaps", len(overlaps),
        flush=True,
    )
    return stats


if __name__ == "__main__":
    os.chdir(ROOT)
    convert()
