#!/usr/bin/env python3
"""Write SPES daily CSVs for the IBM Cloud Code Engine traces.

The unit is an application, NamespaceHash + AppHash. Weeks are unpacked one
at a time and deleted after their per-minute counts are stored. Invocation
lists are histogrammed inside the unpickler so a week of Python floats is
never retained. Other per-request lists (exec times, pod hashes) are dropped.

SPES split: 14 contiguous UTC days starting at the UTC day of the earliest
invocation. Days 1-12 train, days 13-14 test. Trigger is "unknown".
Correlation is not computed.
"""

import datetime
import gc
import json
import math
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pandas as pd
import pickle

from convert_traces import INT32_MAX, MINUTES, N_DAYS, check_written, write_days

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "raw_data/ibm-code-engine")
UNPACK = os.path.join(REPO, "_unpacked")
OUT = os.path.join(ROOT, "raw_data/ibm-spes")
SEVEN = "/opt/homebrew/bin/7z"
N_WEEKS = 10
# A namespace/app column is ~1.8k strings. A pod-hash list is per request.
STRING_KEEP_MAX = 20000
# Unix seconds around 2020-2026 are ~1.6e9. Exec milliseconds are far smaller.
TIME_MIN = 1e8


class FloatRun:
    """One app's timestamps, reduced to per-minute counts.

    List-form pickles fill minute_counts. Numpy float arrays fill bincount,
    whose index 0 is absolute minute bincount_origin.
    """

    __slots__ = ("n", "first", "min_v", "max_v", "minute_counts", "divisor", "bincount", "bincount_origin")

    def __init__(self):
        self.n = 0
        self.first = None
        self.min_v = None
        self.max_v = None
        self.minute_counts = None
        self.divisor = 1.0
        self.bincount = None
        self.bincount_origin = 0

    def add(self, value):
        self.n += 1
        if self.first is None:
            self.first = value
            self.min_v = value
            self.max_v = value
            if value > 1e14:
                self.divisor = 1e9
            elif value > 1e11:
                self.divisor = 1e3
            else:
                self.divisor = 1.0
            self.minute_counts = {}
        else:
            if value < self.min_v:
                self.min_v = value
            if value > self.max_v:
                self.max_v = value
        if value <= 0 or value != value:
            return
        minute = int(math.floor((value / self.divisor) / 60.0))
        day = minute // 1440
        # Keep every day until the trace start is known. After that, keep only
        # the 14-day window. A later week can still move the start earlier.
        origin = _ORIGIN_DAY[0]
        if origin is not None and (day < origin or day >= origin + N_DAYS):
            if day > origin:
                return
        counts = self.minute_counts
        counts[minute] = counts.get(minute, 0) + 1


class StringCount:
    __slots__ = ("n",)

    def __init__(self, n):
        self.n = n


# unix day of the earliest invocation seen so far. None until week 1 finishes.
_ORIGIN_DAY = [None]


class CompactUnpickler(pickle._Unpickler):
    """Unpickle a week frame without retaining per-request lists."""

    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, file):
        super().__init__(file)
        self._dropping_strings = False
        # Memo ids of strings in the current MARK. Nulled if that list is a
        # per-request pod-hash list, so the memo does not keep the week.
        self._pending_string_memos = []

    def load_binfloat(self):
        value = struct.unpack(">d", self.read(8))[0]
        self._push_float(value)

    def load_float(self):
        self._push_float(float(self.readline()[:-1]))

    def _push_float(self, value):
        stack = self.stack
        if stack and isinstance(stack[-1], FloatRun):
            stack[-1].add(value)
            return
        self.append(value)
        if len(self.stack) == 3 and all(isinstance(x, float) for x in self.stack):
            run = FloatRun()
            for item in self.stack:
                run.add(item)
            self.stack = [run]
            self.append = self.stack.append

    def load_short_binunicode(self):
        n = self.read(1)[0]
        self._push_str(str(self.read(n), "utf-8", "surrogatepass"))

    def load_binunicode(self):
        n, = struct.unpack("<I", self.read(4))
        self._push_str(str(self.read(n), "utf-8", "surrogatepass"))

    def load_binunicode8(self):
        n, = struct.unpack("<Q", self.read(8))
        self._push_str(str(self.read(n), "utf-8", "surrogatepass"))

    def _push_str(self, text):
        self.append(text)
        if not self.metastack:
            return
        stack = self.stack
        if self._dropping_strings:
            if len(stack) >= 2 and isinstance(stack[-1], str) and isinstance(stack[-2], StringCount):
                stack[-2].n += 1
                stack.pop()
            return
        if len(stack) == STRING_KEEP_MAX + 1 and all(isinstance(x, str) for x in stack):
            for idx in self._pending_string_memos:
                self.memo[idx] = None
            self._pending_string_memos = []
            self.stack = [StringCount(len(stack))]
            self.append = self.stack.append
            self._dropping_strings = True

    def load_appends(self):
        self._dropping_strings = False
        pickle._Unpickler.load_appends(self)

    def load_append(self):
        self._dropping_strings = False
        pickle._Unpickler.load_append(self)

    def load_reduce(self):
        args = self.stack.pop()
        func = self.stack[-1]
        value = func(*args)
        value = _compact_array(value)
        self.stack[-1] = value
        _drop_big_memo(self.memo)

    def load_build(self):
        pickle._Unpickler.load_build(self)
        top = _compact_array(self.stack[-1])
        self.stack[-1] = top
        _drop_big_memo(self.memo)

    def load_mark(self):
        pickle._Unpickler.load_mark(self)
        self._pending_string_memos = []

    def load_memoize(self):
        top = self.stack[-1]
        idx = len(self.memo)
        # Floats inside list construction are not read back. Keeping them in
        # the memo would retain every timestamp. Strings are kept unless this
        # list is collapsed as a pod-hash run.
        if isinstance(top, float) and self.metastack:
            self.memo[idx] = None
        else:
            self.memo[idx] = top
            if isinstance(top, str) and self.metastack:
                self._pending_string_memos.append(idx)


def _bind(name, opcode):
    CompactUnpickler.dispatch[opcode[0]] = getattr(CompactUnpickler, name)


_bind("load_binfloat", pickle.BINFLOAT)
_bind("load_float", pickle.FLOAT)
_bind("load_short_binunicode", pickle.SHORT_BINUNICODE)
_bind("load_binunicode", pickle.BINUNICODE)
_bind("load_binunicode8", pickle.BINUNICODE8)
_bind("load_appends", pickle.APPENDS)
_bind("load_append", pickle.APPEND)
_bind("load_mark", pickle.MARK)
_bind("load_memoize", pickle.MEMOIZE)
_bind("load_reduce", pickle.REDUCE)
_bind("load_build", pickle.BUILD)


def _drop_big_memo(memo):
    """Drop week-sized buffers. The histogram already owns the counts."""
    for key, value in list(memo.items()):
        if isinstance(value, (bytes, bytearray)) and len(value) > 1_000_000:
            memo[key] = None
        elif isinstance(value, np.ndarray) and value.nbytes > 1_000_000:
            memo[key] = None


def _compact_array(obj):
    """Histogram a per-app float64 timestamp/exec array; drop huge int arrays."""
    if not isinstance(obj, np.ndarray):
        return obj
    if obj.dtype.kind in "iu" and obj.size > 100_000:
        return StringCount(int(obj.size))
    if obj.dtype.kind != "f" or obj.size < 1:
        return obj
    flat = np.asarray(obj, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    run = FloatRun()
    run.n = int(finite.size)
    if finite.size == 0:
        return run
    run.first = float(finite[0])
    run.min_v = float(finite.min())
    run.max_v = float(finite.max())
    # This trace stores seconds (millisecond resolution), not Unix time.
    minutes = np.floor(finite / 60.0).astype(np.int64, copy=False)
    mn = int(minutes.min())
    mx = int(minutes.max())
    if mn < 0 or mx > 2_000_000 or mx - mn > 200_000:
        uniq, cnt = np.unique(minutes, return_counts=True)
        if uniq.size > 200_000:
            raise SystemExit("float array minute span %s..%s has %s distinct minutes" % (mn, mx, uniq.size))
        run.minute_counts = {int(m): int(c) for m, c in zip(uniq, cnt)}
        return run
    shifted = minutes - mn
    run.bincount = np.bincount(shifted)
    run.bincount_origin = mn
    return run


def _cell_counts(cell):
    """Return (minute->count, n_events, first, min_v, max_v) for one cell."""
    if isinstance(cell, FloatRun):
        if cell.bincount is not None and cell.bincount.size:
            origin = int(cell.bincount_origin)
            nz = np.flatnonzero(cell.bincount)
            counts = {origin + int(i): int(cell.bincount[i]) for i in nz}
        else:
            counts = cell.minute_counts or {}
        return counts, cell.n, cell.first, cell.min_v, cell.max_v
    if isinstance(cell, float):
        if cell != cell or cell <= TIME_MIN:
            return {}, 0 if cell != cell else 1, cell, cell, cell
        minute = int(math.floor(cell / 60.0))
        return {minute: 1}, 1, cell, cell, cell
    if isinstance(cell, list):
        counts = {}
        n = 0
        first = None
        min_v = None
        max_v = None
        for value in cell:
            if isinstance(value, FloatRun):
                sub, sn, sf, smin, smax = _cell_counts(value)
                n += sn
                for m, c in sub.items():
                    counts[m] = counts.get(m, 0) + c
                first = sf if first is None else first
                if smin is not None:
                    min_v = smin if min_v is None else min(min_v, smin)
                    max_v = smax if max_v is None else max(max_v, smax)
                continue
            if isinstance(value, np.ndarray):
                value = _compact_array(value)
                sub, sn, sf, smin, smax = _cell_counts(value)
                n += sn
                for m, c in sub.items():
                    counts[m] = counts.get(m, 0) + c
                first = sf if first is None else first
                if smin is not None:
                    min_v = smin if min_v is None else min(min_v, smin)
                    max_v = smax if max_v is None else max(max_v, smax)
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            n += 1
            if first is None:
                first = value
                min_v = value
                max_v = value
            else:
                if value < min_v:
                    min_v = value
                if value > max_v:
                    max_v = value
            if value <= TIME_MIN:
                continue
            minute = int(math.floor(float(value) / 60.0))
            counts[minute] = counts.get(minute, 0) + 1
        return counts, n, first, min_v, max_v
    if cell is None or (isinstance(cell, StringCount)):
        return {}, 0, None, None, None
    raise SystemExit("unexpected invocation cell %s" % type(cell).__name__)


def _iso_day(day_number):
    return datetime.datetime.fromtimestamp(day_number * 86400, datetime.timezone.utc).date().isoformat()


def _free_gb():
    return shutil.disk_usage(ROOT).free / (1024 ** 3)


def _extract(archive, dest):
    if os.path.isdir(dest):
        ready = []
        for dirpath, _dirs, files in os.walk(dest):
            for name in files:
                if name.endswith(".pickle"):
                    ready.append(os.path.join(dirpath, name))
        if len(ready) == 1 and os.path.getsize(ready[0]) > 0:
            print("reuse", ready[0], flush=True)
            return
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    subprocess.check_call([SEVEN, "x", "-bd", "-y", "-o" + dest, archive], cwd=REPO)


def _load_frame(path):
    with open(path, "rb") as f:
        try:
            return CompactUnpickler(f).load()
        except Exception as exc:
            raise SystemExit("compact unpickle failed for %s: %s" % (path, exc)) from exc


def _as_pairs(df):
    """Yield (namespace, app, num_events, invocation_cell) from the real columns."""
    cols = [str(c) for c in df.columns]
    print("columns", cols, "rows", len(df), flush=True)
    rename = {c: c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    def need(*names):
        for name in names:
            if name in df.columns:
                return name
            if name.lower() in lower:
                return lower[name.lower()]
        raise SystemExit("missing %s in %s" % (names, cols))
    ns_c = need("NamespaceHash")
    app_c = need("AppHash")
    if "InvocationTimes" not in df.columns and "invocationtimes" not in lower:
        sample = {}
        for c in df.columns:
            val = df[c].iloc[0]
            sample[str(c)] = type(val).__name__
        raise SystemExit("no InvocationTimes column; cell types %s" % sample)
    inv_c = need("InvocationTimes")
    num_c = None
    if "NumEvents" in df.columns or "numevents" in lower:
        num_c = need("NumEvents")
    # Show the first invocation cell so a schema change is visible in the log.
    probe = df[inv_c].iloc[0]
    if isinstance(probe, FloatRun):
        print(
            "invocation cell FloatRun n", probe.n,
            "first", probe.first, "min", probe.min_v, "max", probe.max_v,
            "divisor", probe.divisor, "minutes", 0 if probe.minute_counts is None else len(probe.minute_counts),
            flush=True,
        )
    else:
        print("invocation cell", type(probe).__name__, flush=True)
    ns = df[ns_c].tolist()
    apps = df[app_c].tolist()
    inv = df[inv_c].tolist()
    nums = df[num_c].tolist() if num_c is not None else [None] * len(df)
    for i in range(len(df)):
        yield str(ns[i]), str(apps[i]), nums[i], inv[i]


def _add_minutes(day_maps, key, counts, max_raw, n_clip):
    apps = day_maps.get(key)
    if apps is None:
        apps = {}
        day_maps[key] = apps
    for minute, count in counts.items():
        if count <= 0:
            continue
        day, slot = divmod(int(minute), 1440)
        arr = apps.get(day)
        if arr is None:
            arr = np.zeros(1440, dtype=np.int32)
            apps[day] = arr
        total = int(arr[slot]) + int(count)
        if total > max_raw[0]:
            max_raw[0] = total
        if total > INT32_MAX:
            n_clip[0] += 1
            total = INT32_MAX
        arr[slot] = np.int32(total)
    return max_raw, n_clip


def _week_path(dest, week):
    matches = []
    for dirpath, _dirs, files in os.walk(dest):
        for name in files:
            if name.endswith(".pickle"):
                matches.append(os.path.join(dirpath, name))
    if len(matches) != 1:
        raise SystemExit("expected one pickle in %s, found %s" % (dest, matches))
    return matches[0]


def _consume_week(df, day_maps, all_apps, max_raw, n_clip, mismatches):
    rows = 0
    events = 0
    min_minute = None
    max_minute = None
    for ns, app, num, cell in _as_pairs(df):
        rows += 1
        key = (ns, app)
        all_apps.add(key)
        counts, n, first, min_v, max_v = _cell_counts(cell)
        events += n
        if num is not None and int(num) != n:
            mismatches.append((ns, app, int(num), n, first, min_v, max_v))
        if counts:
            lo = min(counts)
            hi = max(counts)
            min_minute = lo if min_minute is None else min(min_minute, lo)
            max_minute = hi if max_minute is None else max(max_minute, hi)
            _add_minutes(day_maps, key, counts, max_raw, n_clip)
    return {
        "rows": rows,
        "events_histogrammed": events,
        "min_unix_minute": min_minute,
        "max_unix_minute": max_minute,
    }


def _build_grids(day_maps, origin):
    grids = {}
    meta = {}
    window = list(range(origin, origin + N_DAYS))
    for (ns, app), days in day_maps.items():
        grid = np.zeros(MINUTES, dtype=np.int32)
        used = False
        for day, arr in days.items():
            slot = day - origin
            if slot < 0 or slot >= N_DAYS:
                continue
            if arr.any():
                used = True
            grid[slot * 1440:(slot + 1) * 1440] = arr
        if not used:
            continue
        # Fixed-width hashes concatenate. Otherwise keep both fields readable.
        if len(ns) == len(app) and "\t" not in ns and "\t" not in app and "/" not in ns and "/" not in app:
            func = ns + app
            key_rule = "NamespaceHash+AppHash concatenated (equal length)"
        else:
            func = ns + "/" + app
            key_rule = "NamespaceHash/AppHash"
        grids[func] = grid
        meta[func] = (ns, app, func, "unknown")
    return grids, meta, window, key_rule


def self_check():
    _ORIGIN_DAY[0] = None
    df = pd.DataFrame({
        "NamespaceHash": ["ns1", "ns2"],
        "AppHash": ["app1", "app2"],
        "NumEvents": [4, 1],
        "InvocationTimes": [[1680000000.1, 1680000060.2, 1680000120.3, 1680000180.4], [1680001000.0]],
        "AppExecTimes": [[10.0, 11.0, 12.0], [5.0]],
        "TotalExecTimes": [[20.0, 21.0, 22.0], [9.0]],
        "PodHash": [["p1", "p2", "p3"], ["q"] * (STRING_KEEP_MAX + 1)],
    })
    path = "/tmp/spes_ibm_self.pickle"
    df.to_pickle(path)
    loaded = _load_frame(path)
    got = list(_as_pairs(loaded))
    counts, n, _f, _a, _b = _cell_counts(got[0][3])
    assert n == 4, n
    assert counts == {28000000: 1, 28000001: 1, 28000002: 1, 28000003: 1}, counts
    counts2, n2, _f, _a, _b = _cell_counts(got[1][3])
    assert n2 == 1 and counts2 == {28000016: 1}, (n2, counts2)
    # Exec times must not be parsed as invocations: they live in another column.
    os.remove(path)
    df = pd.DataFrame({
        "NamespaceHash": ["ns"],
        "AppHash": ["app"],
        "NumEvents": [3],
        "InvocationTimes": [np.array([18000.1, 18060.4, 18120.0])],
        "AppExecTimes": [np.array([10.0, 65.0, 118.0])],
        "TotalExecTimes": [np.array([20.0, 21.0, 22.0])],
        "PodHash": [np.array([1, 2, 3], dtype=np.int64)],
    })
    path = "/tmp/spes_ibm_self_np.pickle"
    df.to_pickle(path)
    loaded = _load_frame(path)
    got = list(_as_pairs(loaded))
    counts, n, _f, _a, _b = _cell_counts(got[0][3])
    assert n == 3, n
    assert counts == {300: 1, 301: 1, 302: 1}, counts
    os.remove(path)
    print("ibm self-check ok", flush=True)


def main():
    self_check()
    if not os.path.isdir(os.path.join(REPO, "compressed_data")):
        raise SystemExit("IBM repo is not checked out at %s" % REPO)
    _ORIGIN_DAY[0] = None
    day_maps = {}
    all_apps = set()
    max_raw = [0]
    n_clip = [0]
    mismatches = []
    week_stats = []
    # Configurations are tiny and name every app the trace was published with.
    cfg_archive = os.path.join(REPO, "compressed_data", "app_configs.7z")
    cfg_apps = None
    if os.path.isfile(cfg_archive):
        dest = os.path.join(UNPACK, "configs")
        _extract(cfg_archive, dest)
        cfg_path = _week_path(dest, 0)
        cfg = pd.read_pickle(cfg_path)
        print("app_configs columns", list(cfg.columns), "rows", len(cfg), flush=True)
        if "NamespaceHash" in cfg.columns and "AppHash" in cfg.columns:
            cfg_apps = set(zip(cfg["NamespaceHash"].astype(str), cfg["AppHash"].astype(str)))
        shutil.rmtree(dest, ignore_errors=True)
        del cfg
        gc.collect()
    for week in range(1, N_WEEKS + 1):
        archive = os.path.join(REPO, "compressed_data", "week_%d.7z.001" % week)
        if not os.path.isfile(archive):
            raise SystemExit("missing %s" % archive)
        free = _free_gb()
        print("week", week, "free_gb", round(free, 1), flush=True)
        if free < 12:
            raise SystemExit("refusing to unpack week %s with only %.1f GB free" % (week, free))
        dest = os.path.join(UNPACK, "week_%d" % week)
        _extract(archive, dest)
        path = _week_path(dest, week)
        print("load", path, "bytes", os.path.getsize(path), flush=True)
        df = _load_frame(path)
        if not isinstance(df, pd.DataFrame):
            raise SystemExit("week %s unpickled %s, not a DataFrame" % (week, type(df).__name__))
        before = len(mismatches)
        stats = _consume_week(df, day_maps, all_apps, max_raw, n_clip, mismatches)
        if len(mismatches) > before:
            print("numevents mismatches", len(mismatches) - before, "first", mismatches[before:before + 3], flush=True)
            raise SystemExit("InvocationTimes histogram does not match NumEvents")
        stats["week"] = week
        stats["pickle_bytes"] = os.path.getsize(path)
        week_stats.append(stats)
        print("week", week, "done", stats, "apps", len(all_apps), flush=True)
        del df
        gc.collect()
        shutil.rmtree(dest, ignore_errors=True)
        if len(week_stats) >= 2 and stats["min_unix_minute"] is not None:
            prev_min = week_stats[-2]["min_unix_minute"]
            if prev_min is not None and stats["min_unix_minute"] < prev_min + 1000:
                raise SystemExit(
                    "week %s timestamps restart at minute %s; previous week min %s"
                    % (week, stats["min_unix_minute"], prev_min)
                )
        if _ORIGIN_DAY[0] is None and stats["min_unix_minute"] is not None:
            _ORIGIN_DAY[0] = stats["min_unix_minute"] // 1440
            print("origin day", _ORIGIN_DAY[0], flush=True)
        elif stats["min_unix_minute"] is not None and stats["min_unix_minute"] // 1440 < _ORIGIN_DAY[0]:
            _ORIGIN_DAY[0] = stats["min_unix_minute"] // 1440
            print("origin moved earlier", _ORIGIN_DAY[0], flush=True)
    if _ORIGIN_DAY[0] is None:
        raise SystemExit("no invocation minutes in any week")
    if mismatches:
        print("numevents mismatches", len(mismatches), "first", mismatches[:3], flush=True)
        # A mismatch means the histogram did not see the published event count.
        raise SystemExit("InvocationTimes histogram does not match NumEvents")
    origin = _ORIGIN_DAY[0]
    # Minute indexes above this are Unix days (sec/86400). The published values
    # are seconds from the trace zero, a few thousand, so the window is not a
    # calendar date.
    unix_clock = origin > 10000
    grids, meta, window, key_rule = _build_grids(day_maps, origin)
    if not grids:
        raise SystemExit("14-day window has no application invocations")
    expected = int(sum(int(g.sum(dtype=np.int64)) for g in grids.values()))
    test_sum = int(sum(int(g[12 * 1440:].sum(dtype=np.int64)) for g in grids.values()))
    write_days(grids, meta, OUT)
    written = check_written(OUT, expected)
    in_window_apps = len(grids)
    stats = {
        "dataset": "ibm-cloud-code-engine",
        "unit": "application (NamespaceHash + AppHash)",
        "apps_in_app_configs": None if cfg_apps is None else len(cfg_apps),
        "apps_in_full_trace": len(all_apps),
        "apps_in_14day_window": in_window_apps,
        "functions": in_window_apps,
        "weeks_processed": [w["week"] for w in week_stats],
        "weeks": week_stats,
        "clock": "unix" if unix_clock else "seconds from the trace zero, not Unix epoch",
        "origin_day_index": int(origin),
        "origin_label": _iso_day(origin) if unix_clock else "minute %s" % (origin * 1440),
        "window_labels": [_iso_day(d) for d in window] if unix_clock else [
            "d%02d minutes %s-%s" % (i + 1, d * 1440, (d + 1) * 1440 - 1) for i, d in enumerate(window)
        ],
        "window": (
            "14 contiguous UTC days starting at the UTC day of the earliest invocation"
            if unix_clock else
            "14 contiguous trace-days of 1440 minutes from the earliest invocation, aligned down to the trace-day grid"
        ),
        "train_days": "d01-d12",
        "test_days": "d13-d14",
        "max_minute_count": int(max_raw[0]),
        "int32_clipped_increments": int(n_clip[0]),
        "written_invocation_sum": written,
        "test_day_invocation_sum": test_sum,
        "function_key": key_rule,
        "owner": "NamespaceHash",
        "application": "AppHash",
        "trigger": "unknown",
        "minute": "floor(InvocationTimes seconds / 60); other per-request columns are not counted",
        "correlation": "disabled",
        "func_candi_json": "not read",
        "numevents_mismatches": len(mismatches),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "convert_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        "ibm apps_full", len(all_apps),
        "apps_window", in_window_apps,
        "configs", stats["apps_in_app_configs"],
        "invocations", written,
        "test_invocations", test_sum,
        "max_minute", max_raw[0],
        "clipped", n_clip[0],
        flush=True,
    )
    if n_clip[0]:
        print("CLIPPED minute counts to int32; unclipped max was", max_raw[0], flush=True)
    return stats


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        main()
