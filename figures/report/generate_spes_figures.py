"""Regenerate the report figures from SPES outputs and Huawei trace archives."""

import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
INK = "#18283B"
MUTED = "#4F6073"
GRID = "#DCE3EB"
BLUE = "#205A91"
PALE_BLUE = "#D6E7F4"
ORANGE = "#A84620"
SLATE = "#516378"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
})


def load_counts(run):
    result = ROOT / "result"
    if run == "azure2019":
        paths = result / "func_invok_rebuilt.json", result / "func_cold_08-23-43.json"
    else:
        folder = result / "runs" / run
        paths = folder / "func_invok.json", folder / "func_cold.json"
    invocations, cold = [json.loads(path.read_text()) for path in paths]
    assert invocations.keys() == cold.keys()
    assert all(0 <= cold[k] <= v and v > 0 for k, v in invocations.items())
    return invocations, cold


def q3(invocations, cold, minimum=1):
    rates = [cold.get(k, 0) / v for k, v in invocations.items() if v >= minimum]
    return float(np.percentile(rates, 75)), len(rates)


def percent(value):
    return f"{100 * value:.2f}".rstrip("0").rstrip(".") + "%"


def heading(fig, title, subtitle):
    fig.text(.055, .955, title, va="top", fontsize=22, weight="bold")
    fig.text(.055, .878, subtitle, va="top", fontsize=11, color=MUTED)


def footer(fig, *lines):
    fig.text(.055, .04, "\n".join(lines), va="bottom", fontsize=9,
             color=MUTED, linespacing=1.5)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def format_axis(ax, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=.8)
    scale = ax.xaxis if axis == "x" else ax.yaxis
    scale.set_major_formatter(PercentFormatter(xmax=1, decimals=0))


def comparison(runs, reported):
    fig = plt.figure(figsize=(11.2, 5.35))
    heading(fig, "SPES cold-start rates across traces",
            "Q3 is the 75th percentile of per-unit cold-start rates over two test days.")
    ax = fig.add_axes([.29, .25, .40, .52])
    format_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-.025, 1.065)
    ax.set_ylim(5.65, -.65)
    ax.set_yticks([])
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel("Q3 cold-start rate", labelpad=9)

    specs = [
        ("paper", "Azure 2019 · paper", "Published SPES result"),
        ("azure2019", "Azure 2019 · replay", "Same app or same owner"),
        ("azure2021_corr", "Azure 2021 · replay", "Same app only"),
        ("huawei_all", "Huawei 2025 · replay", "None · five regions"),
        ("ibm", "IBM Code Engine · replay", "None · units are apps"),
        ("globus", "Globus Compute · replay", "None · exploratory subset"),
    ]
    ax.text(1.11, 1.045, "Offline correlation scope", transform=ax.transAxes,
            weight="bold", fontsize=11)
    for i, (key, name, detail) in enumerate(specs):
        report_key = "globus_half" if key == "globus" else key
        expected = reported[report_key]
        if key == "paper":
            val, count = float(expected["q3_csr"]), int(expected["functions_simulated"])
        else:
            val, count = q3(*runs[key])
            assert round(val, 4) == float(expected["q3_csr"])
            assert count == int(expected["functions_simulated"])
        color = SLATE if key == "paper" else BLUE
        ax.text(-.15, i - .10, name, transform=ax.get_yaxis_transform(),
                ha="right", va="center", weight="bold", fontsize=10.6)
        ax.text(-.15, i + .16, f"n = {count:,}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", color=MUTED, fontsize=9)
        ax.text(1.11, i, detail, transform=ax.get_yaxis_transform(), va="center",
                fontsize=10.6)
        ax.hlines(i, 0, val, color=color, linewidth=2)
        ax.scatter(val, i, s=49, marker="D" if key == "paper" else "o",
                   edgecolors=color, facecolors="white" if key == "paper" else color,
                   linewidths=1.6, zorder=3)
        ax.annotate(percent(round(val, 4)), (val, i), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=10.6,
                    weight="bold", color=color)
    footer(fig,
           "Each row uses its own test cohort. Correlation settings and units differ across traces.",
           "Sources: SPES (ICDE 2024, Fig. 8 discussion) and local per-unit simulation outputs.")
    save(fig, "01_spes_q3_across_traces")


def recorded_huawei(invocations):
    totals = {metric: Counter() for metric in ("requests", "num_cold_starts")}
    root = ROOT / "raw_data/huawei-2025"
    for region in range(1, 6):
        with ZipFile(root / f"region{region}_quantiles.zip") as archive:
            for day in (12, 13):
                for metric, counts in totals.items():
                    with archive.open(f"R{region}/{metric}/day_{day:02d}.csv") as member:
                        data = pd.read_csv(member)
                    assert np.array_equal(data["time"], day * 86400 + np.arange(1440) * 60)
                    values = data.drop(columns=["day", "time"]).fillna(0)
                    assert (values.to_numpy() >= 0).all()
                    counts.update(values.sum(axis=0).astype("int64").to_dict())
    assert {k: v for k, v in totals["requests"].items() if v} == invocations
    cold = totals["num_cold_starts"]
    assert all(k in invocations for k, count in cold.items() if count > 0)
    return cold


def huawei_graph(invocations, simulated, recorded):
    fig = plt.figure(figsize=(11.2, 4.7))
    heading(fig, "Huawei 2025: simulated vs. recorded cold starts",
            "Same five regions, same 6,504 function IDs, same two test days.")
    ax = fig.add_axes([.245, .29, .59, .39])
    ax.set_xlim(0, 1)
    ax.set_ylim(1.55, -.55)
    ax.set_yticks([0, 1], ["SPES replay", "Huawei logs"], fontweight="bold")
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel("Share of test functions, sorted by cold-start rate", labelpad=9)
    format_axis(ax)
    ax.spines["left"].set_visible(False)
    labels = ["Zero cold starts", "Above 0%, below 100%", "100% or higher"]
    colors = [PALE_BLUE, BLUE, ORANGE]
    for i, data in enumerate([simulated, recorded]):
        n_zero = sum(data.get(k, 0) == 0 for k in invocations)
        n_high = sum(data.get(k, 0) >= v for k, v in invocations.items())
        counts = [n_zero, len(invocations) - n_zero - n_high, n_high]
        value, n = q3(invocations, data)
        assert (n_zero, sum(data.values()), value) == [(5068, 12498, 0.0), (235, 801435, 1.0)][i]
        left = 0
        for j, (count, color, label) in enumerate(zip(counts, colors, labels)):
            share = count / n
            ax.barh(i, share, left=left, height=.46, color=color,
                    edgecolor="white", linewidth=.8, label=label if i == 0 else None)
            if share >= .08:
                ax.text(left + share / 2, i, f"{share:.1%}", ha="center", va="center",
                        color=INK if j == 0 else "white", weight="bold", fontsize=11)
            left += share
        ax.text(1.055, i - .08, f"Q3 = {percent(value)}", transform=ax.get_yaxis_transform(),
                weight="bold", va="center", fontsize=12)
        ax.text(1.055, i + .17, f"{sum(data.values()):,} cold starts",
                transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=9)
    ax.axvline(.75, color=INK, linewidth=1.15, linestyle=(0, (4, 3)), zorder=3)
    ax.legend(loc="lower center", bbox_to_anchor=(.49, 1.06), ncol=3, frameon=False,
              fontsize=9.5, columnspacing=1.55, handlelength=1.3)
    above_one = sum(recorded.get(k, 0) > v for k, v in invocations.items())
    footer(fig,
           "Zero-cold share: 77.9% in the replay vs. 3.6% in the logs. The dashed line marks the 75th percentile.",
           f"Recorded counts and request totals are aggregated separately. The ratio exceeds 100% for {above_one} function IDs.",
           "Sources: Huawei Public Cloud Trace 2025, days 12–13, requests and num_cold_starts; SPES five-region replay.")
    save(fig, "02_huawei_replay_vs_recorded")


def sensitivity(runs):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.95), sharey=True)
    fig.subplots_adjust(left=.10, right=.955, bottom=.27, top=.74, wspace=.18)
    heading(fig, "Q3 by minimum request count",
            "Each point recalculates Q3 after requiring a minimum number of requests per test unit.")
    thresholds = [1, 2, 5, 10]
    x = np.arange(len(thresholds))
    specs = [
        ("azure2021_corr", "Azure 2021", BLUE, "o"),
        ("huawei_all", "Huawei 2025", "#91601A", "s"),
        ("ibm", "IBM Code Engine", SLATE, "D"),
    ]
    for ax, (key, name, color, marker) in zip(axes, specs):
        values, ns = zip(*(q3(*runs[key], minimum) for minimum in thresholds))
        format_axis(ax, "y")
        ax.set_title(name, loc="left", fontsize=13, weight="bold", pad=14)
        ax.set_ylim(-.025, .80)
        ax.set_yticks([0, .2, .4, .6, .8])
        ax.set_xlim(-.37, 3.38)
        ax.set_xticks(x, [f"≥{t}\n{n:,}" for t, n in zip(thresholds, ns)])
        ax.tick_params(axis="x", length=0, pad=8)
        ax.plot(x, values, marker=marker, markersize=6.5, linewidth=2, color=color)
        for xi, value in zip(x, values):
            offset = (7, 7) if key == "azure2021_corr" and xi in (1, 2) else (0, 9)
            ax.annotate(percent(value), (xi, value), xytext=offset,
                        textcoords="offset points", ha="center", va="bottom",
                        color=color, weight="bold", fontsize=9.5)
    axes[0].set_ylabel("Q3 cold-start rate", labelpad=9)
    fig.text(.52, .135, "Minimum test requests · second row shows remaining units",
             ha="center", color=MUTED, fontsize=10.5)
    footer(fig,
           "SPES simulations. Azure 2021 uses its same-app offline map. Huawei and IBM have empty offline maps.",
           "The thresholds change which units enter the percentile. The underlying runs stay the same.")
    save(fig, "03_q3_by_minimum_requests")


def main():
    keys = ["azure2019", "azure2021_corr", "huawei_all", "ibm", "globus"]
    runs = {key: load_counts(key) for key in keys}
    with (ROOT / "SPES_ALL_RUNS_REPORT.csv").open() as f:
        reported = {row["run_id"]: row for row in csv.DictReader(f)}
    comparison(runs, reported)
    requests, simulated = runs["huawei_all"]
    huawei_graph(requests, simulated, recorded_huawei(requests))
    sensitivity(runs)


if __name__ == "__main__":
    main()
