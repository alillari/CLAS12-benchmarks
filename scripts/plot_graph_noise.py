#!/usr/bin/env python3
"""Plot held-out cluster-noise curves and occupancy results from a completed run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"gravnet": "#136b81", "garnet": "#ac4a17", "sage": "#6653a1",
           "fm_adapter": "#3c7851", "fm_frozen": "#aa3767"}
MODEL_LABELS = {"gravnet": "GravNet", "garnet": "GarNet", "sage": "SAGEConv",
                "fm_adapter": "FM4NPP AdapterOnly", "fm_frozen": "Frozen FM4NPP"}


def plot(root, destination):
    summary = json.loads((root / "summary.json").read_text())
    if not summary.get("bootstrap"):
        raise ValueError("Expected a completed held-out evaluation")
    destination.mkdir(parents=True, exist_ok=False)
    fig, (tradeoff, pr, roc) = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    fig_occupancy, ax_occupancy = plt.subplots(figsize=(7.8, 4.4), layout="constrained")
    group_names = ("occupancy_0_99", "occupancy_100_149", "occupancy_150_199",
                   "occupancy_200_249", "occupancy_250_inf")
    for name, arm in summary["arms"].items():
        row = arm["test"]
        model = arm["arm"]["model"]["architecture"]
        label = MODEL_LABELS[model]
        color = PALETTE[model]
        with np.load(root / name / "test_curves.npz") as curve:
            tradeoff.plot(curve["signal_retention"], curve["noise_rejection"], color=color, label=label)
            pr.plot(curve["signal_recall"], curve["signal_precision"], color=color,
                    label=f"{label}  AP {row['signal_average_precision']:.3f}")
            roc.plot(curve["false_signal_rate"], curve["signal_retention"], color=color,
                     label=f"{label}  AUC {row['signal_roc_auc']:.3f}")
        groups = json.loads((root / name / "test.json").read_text())["groups"]
        points = [groups.get(group, {}).get("operating_points", {}).get("retain_99pct", {})
                  for group in group_names]
        ax_occupancy.plot(range(len(points)), [p.get("noise_rejection", np.nan) for p in points],
                          marker="o", color=color, label=label)
    baseline = row["coatjava_unassigned"]
    tradeoff.scatter(baseline["signal_retention"], baseline["noise_rejection"],
                     marker="D", color="black", s=28, label="COATJAVA unassigned")
    prevalence = row["signal_clusters"] / (row["signal_clusters"] + row["noise_clusters"])
    pr.axhline(prevalence, color=".55", linewidth=1, linestyle="--", label="Signal prevalence")
    roc.plot([0, 1], [0, 1], color=".55", linewidth=1, linestyle="--")
    tradeoff.set(xlabel="Signal retention", ylabel="Noise rejection", xlim=(.90, 1.001), ylim=(0, 1.02))
    pr.set(xlabel="Signal recall", ylabel="Signal precision", xlim=(0, 1.01), ylim=(0, 1.02))
    roc.set(xlabel="False signal rate", ylabel="Signal retention", xlim=(0, 1.01), ylim=(0, 1.02))
    for axis in (tradeoff, pr, roc, ax_occupancy):
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("CLAS12 CVT cluster noise tagging on the held-out v2 cohort")
    fig.savefig(destination / "noise_curves.png", dpi=180)
    ax_occupancy.set(xlabel="Clusters per event", ylabel="Noise rejection",
                     title="Validation threshold targeting 99% overall signal retention",
                     xticks=list(range(len(group_names))), xticklabels=["0–99", "100–149", "150–199", "200–249", "≥250"],
                     ylim=(0, 1.02))
    fig_occupancy.savefig(destination / "noise_by_occupancy.png", dpi=180)
    plt.close("all")


def plot_budgets(roots, destination):
    series = {}
    shared = set()
    for root in roots:
        report = json.loads((root / "summary.json").read_text())
        shared.add(report["cohort_sha256"])
        budget = report["settings"]["train"]
        for arm in report["arms"].values():
            name = arm["arm"]["model"]["architecture"]
            scores = arm["test"]
            value = scores["operating_points"]["retain_99pct"]["noise_rejection"]
            series.setdefault(name, []).append((budget, value, scores["signal_roc_auc"]))
    if len(shared) != 1:
        raise ValueError("Label-budget results require a single shared cohort")
    destination.mkdir(parents=True, exist_ok=False)
    fig, (retention, roc) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for name, rows in series.items():
        budget, rejection, auc = zip(*sorted(rows), strict=True)
        for axis, values in ((retention, rejection), (roc, auc)):
            axis.plot(budget, values, color=PALETTE[name], marker="o", label=MODEL_LABELS[name])
    retention.set_ylabel("Noise rejection at validation 99% signal retention")
    roc.set_ylabel("Signal ROC AUC")
    budgets = sorted({row[0] for rows in series.values() for row in rows})
    for axis in (retention, roc):
        axis.set(xscale="log", xlabel="Labeled training events",
                 xticks=budgets, xticklabels=[f"{k:,}" for k in budgets])
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("The 12,000-event fits use a longer update budget")
    fig.savefig(destination / "noise_by_label_budget.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label-budget", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.label_budget:
        plot_budgets([args.run, *args.label_budget], args.output)
    else:
        plot(args.run, args.output)
