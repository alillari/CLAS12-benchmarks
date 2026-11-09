"""Plot an external frozen Exa.TrkX GNN-cut validation sweep."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.input)
    output = Path(args.output) if args.output else root
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "summary.json").read_text())
    curve = json.loads((root / "threshold_curve.json").read_text())
    x = [r["cut"] for r in curve]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, [r["track_f1_global"] for r in curve], label="Track F1", color="#2463a6")
    for key, label, color in (("track_purity_global", "Track purity", "#db8535"),
                              ("track_efficiency_global", "Track efficiency", "#31856a")):
        ax.plot(x, [r["metrics"][key] for r in curve], label=label, color=color)
    ax.set(title="Matched-track metrics", ylabel="Fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[0, 1]
    ax.plot(x, [r["metrics"]["ari_signal"] for r in curve], color="#2463a6", label="Signal ARI")
    ax.set(title="Signal clustering", ylabel="Adjusted Rand index")
    ax = axes[1, 0]
    ax.plot(x, [r["graph"]["fully_connected_track_fraction"] for r in curve],
            label="All true hits connected", color="#2463a6")
    ax.plot(x, [r["clean_fully_connected_track_fraction"] for r in curve],
            label="Connected, no other signal track", color="#31856a")
    ax.set(title="True-track connectivity", ylabel="Fraction of true tracks")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1, 1]
    for key, label, color in (("merge_rate", "Merged predictions (event mean)", "#ac4b4b"),
                              ("signal_loss_to_background", "Signal hits lost (event mean)", "#db8535")):
        ax.plot(x, [r["metrics"][key] for r in curve], label=label, color=color)
    ax.plot(x, [r["merge_exposed_true_track_fraction"] for r in curve],
            label="True tracks touching a merge (pooled)", color="#765594")
    ax.set(title="Merges and rejected signal", ylabel="Fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    saved, best = summary["saved_cut"], summary["best_by_f1"]["cut"]
    for ax in axes.flat:
        ax.axvline(saved, color="0.35", linestyle=":", label=f"Saved cut {saved:g}")
        if best != saved:
            ax.axvline(best, color="0.35", linestyle="--", label=f"Best validation F1 cut {best:g}")
        ax.set(xlabel="GNN probability cut (retain score ≥ cut)", xlim=(0, 1), ylim=(0, 1.025))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Frozen Exa.TrkX · {summary['events']:,} validation events · only the GNN cut varies", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"gnn_threshold_sweep.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
