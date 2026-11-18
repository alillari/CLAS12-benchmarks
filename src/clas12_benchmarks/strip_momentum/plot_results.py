"""Plot the holdout comparison by true momentum on matched test tracks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("sensitivity", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.metrics.read_text())
    all_bins = result["differential"]["momentum_GeV_c"]
    trimmed_bins = json.loads(args.sensitivity.read_text())["thresholds"]["20"]["momentum_bins"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": "#202b36",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#939fab",
        "axes.titleweight": "bold", "xtick.color": "#455466", "ytick.color": "#455466",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 5.0))
    palette = (("#116878", "o"), ("#a8442c", "s"))
    series = (("adapter_paired", "Strip AdapterOnly"), ("coatjava_paired", "COATJAVA"))
    for axis, (bins, key, factor, label, scope) in zip(axes, (
        (trimmed_bins, "p_mae_gev", 1000, "Momentum MAE (MeV/c)", "COATJAVA momentum ≤ 20 GeV/c"),
        (all_bins, "relative_c68", 100, "Central 68% half-width of Δp/p (%)", "All finite matches"),
    )):
        names = list(bins)
        counts = [row["coatjava_paired"]["tracks"] for row in bins.values()]
        positions = np.arange(len(bins))
        for (field, legend), (color, marker) in zip(series, palette):
            y = [row[field].get(key, np.nan) * factor if n >= 50 else np.nan
                 for row, n in zip(bins.values(), counts)]
            axis.plot(positions, y, marker=marker, color=color, lw=2, ms=5, label=legend)
        axis.set_title(label, loc="left", fontsize=12.5, pad=24)
        axis.text(0, 1.018, scope, transform=axis.transAxes, fontsize=9.5, color="#455466")
        axis.set_xticks(positions, [f"{name.replace('–inf', '+')}\n{n:,}" for name, n in zip(names, counts)],
                        fontsize=9)
        axis.set_xlabel("True momentum bin (GeV/c)\nMatched tracks per bin", labelpad=10)
        axis.grid(axis="y", color="#dbe2e9", linewidth=.8)
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_locator(MaxNLocator(6))
        axis.tick_params(axis="both", length=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Strip momentum regression on matched test tracks", fontsize=18, x=.08, y=.965,
                 ha="left", fontweight="bold", color="#202b36")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.065, .91), ncols=2,
               frameon=False)
    fig.subplots_adjust(left=.08, right=.97, top=.73, bottom=.21, wspace=.25)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
