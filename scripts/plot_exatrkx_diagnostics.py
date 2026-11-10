"""Plot external frozen Exa.TrkX diagnostic JSON files. Requires Matplotlib."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.input)
    output = Path(args.output) if args.output else root
    output.mkdir(parents=True, exist_ok=True)
    def read(name):
        return json.loads((root / name).read_text())
    summary, radius, ks, distributions = map(read, ("summary.json", "radius_scan.json", "k_scan.json", "distance_distributions.json"))
    config = read("provenance.json")["config"]
    saved_radius, saved_k = config["candidate_radius"], config["candidate_k"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, green = "#2463a6", "#db8535", "#31856a"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    for key, label, color in (("same_track", "Same-track pairs", blue),
                              ("different_signal_tracks", "Different-track signal pairs", orange),
                              ("track_mst_connection_radius", "Per-track MST connection radius", green)):
        curve = distributions[key]["cdf"]
        if curve:
            ax.plot(curve["distance"], curve["fraction"], label=label, color=color)
    ax.axvline(saved_radius, color="0.4", linestyle=":", label=f"Saved candidate radius = {saved_radius:g}")
    ax.set(xlabel="Euclidean embedding distance", ylabel="Cumulative fraction", title="Embedding distance distributions")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8, loc="lower right")
    finite = [row["radius"] for row in radius if row["radius"] is not None]
    finite = finite or [0., float(saved_radius)]
    infinity_x = max(finite) + max(max(finite) - min(finite), 1.) * .12
    x = [infinity_x if row["radius"] is None else row["radius"] for row in radius]
    ax = axes[0, 1]
    for key, label, color in (("true_pair_recall", "True-pair recall", blue),
                              ("fully_connected_track_fraction", "Fully connected true tracks", green)):
        ax.plot(x, [row[key] for row in radius], marker="o", ms=3, color=color, label=label)
    ax.axvline(saved_radius, color="0.4", linestyle=":")
    ticks = [v for v in (0, .5, 1, 1.5, 2, 3, 4) if min(finite) <= v <= max(finite)]
    if any(row["radius"] is None for row in radius):
        ax.set_xticks(ticks + [infinity_x], [str(v) for v in ticks] + ["∞"])
    ax.set(xlabel="Candidate radius (k fixed at saved value)", ylabel="Fraction", title="Candidate recall versus radius", ylim=(0, 1.025))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1, 0]
    maximum_k = max((row["k"] for row in ks if row["k"] is not None), default=saved_k)
    x = [maximum_k * 2 if row["k"] is None else row["k"] for row in ks]
    for key, label, color in (("true_pair_recall", "True-pair recall", blue),
                              ("fully_connected_track_fraction", "Fully connected true tracks", green)):
        ax.plot(x, [row[key] for row in ks], marker="o", ms=3, color=color, label=label)
    ax.set_xscale("log", base=2)
    ticks = [v for v in (1, 2, 4, 8, 16, 32, 64, 128) if v <= maximum_k]
    if any(row["k"] is None for row in ks):
        ax.set_xticks(ticks + [maximum_k * 2], [str(v) for v in ticks] + ["all"])
    ax.axvline(saved_k, color="0.4", linestyle=":")
    ax.set(xlabel="Directed neighbors per point (radius fixed)", ylabel="Fraction", title="Candidate recall versus k", ylim=(0, 1.025))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1, 1]
    stages = ("candidate", "after_filter", "after_gnn")
    labels = ("Candidates", "After filter", "After GNN cut")
    full = np.array([summary["stages"][s]["graph"]["fully_connected_track_fraction"] for s in stages], dtype=float)
    half = np.array([summary["stages"][s]["graph"]["at_least_half_connected_track_fraction"] for s in stages], dtype=float)
    positions = np.arange(3)
    ax.bar(positions - .18, full, .36, label="All track hits connected", color=green)
    ax.bar(positions + .18, half, .36, label="At least half connected", color=blue)
    ax.set(xticks=positions, xticklabels=labels, ylabel="Fraction of true tracks", title="Connectivity through the pipeline", ylim=(0, 1.09))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    for i in range(3):
        for offset, values in ((-.18, full), (.18, half)):
            value = values[i]
            ax.text(i + offset, value + .012 if np.isfinite(value) else .012,
                    f"{100*value:.1f}%" if np.isfinite(value) else "n/a", ha="center", fontsize=8)
    ax.legend(fontsize=8, loc="lower left")
    n_tracks = summary["stages"]["candidate"]["graph"]["tracks"]
    fig.suptitle(f"Frozen Exa.TrkX graph diagnostics · {summary['events']:,} {summary['split']} events · {n_tracks:,} true tracks", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"embedding_and_candidates.{suffix}", dpi=180)
    plt.close(fig)
    columns = (("ari_signal", "Signal ARI"), ("track_purity_global", "Track purity"), ("track_efficiency_global", "Track efficiency"))
    if "mc_track_dice_mean" in summary["actual_gnn"]:
        columns = (("mc_track_dice_mean", "MC track Dice"), ("mc_hit_purity_global", "MC hit purity"),
                   ("mc_hit_completeness_global", "MC hit completeness"),
                   ("track_efficiency_global", "Accepted-track efficiency"))
    fig, axes = plt.subplots(1, len(columns), figsize=(4*len(columns), 4), constrained_layout=True)
    labels = ("Candidate\noracle", "After-filter\noracle", "After-GNN\noracle", "Actual GNN\nreconstruction")
    for ax, (key, title) in zip(axes, columns):
        values = np.array([summary["stages"][s]["oracle"][key] for s in stages] + [summary["actual_gnn"][key]], dtype=float)
        ax.bar(np.arange(4), values, color=[blue, green, orange, "#ac4b4b"])
        ax.set(xticks=np.arange(4), xticklabels=labels, title=title, ylim=(-1 if key == "ari_signal" else 0, 1.1))
        ax.tick_params(axis="x", labelsize=8)
        if key != "ari_signal":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        for i, value in enumerate(values):
            label = (f"{value:.3f}" if key == "ari_signal" else f"{100*value:.1f}%") if np.isfinite(value) else "n/a"
            ax.text(i, value + .015 if np.isfinite(value) else .015, label, ha="center", fontsize=9)
    fig.suptitle("Perfect-edge-label connected-component oracles versus actual reconstruction", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"oracle_reconstruction.{suffix}", dpi=180)
    plt.close(fig)
    print(f"Saved diagnostic PNG and SVG plots under {output}")


if __name__ == "__main__":
    main()
