"""Compare signed momentum and angular residuals on matched test tracks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import numpy as np

from .strip_adapter import errors


PANELS = (
    ("delta_p_over_p", r"$\Delta p/p$ (%)", 25, 10),
    ("delta_theta", r"$\Delta\theta$ (degrees)", 6, 2),
    ("delta_phi", r"$\Delta\phi$ (degrees)", 2, 1),
)
METHODS = ("Strip AdapterOnly", "COATJAVA")
P_EDGES = np.linspace(0, 7.3, 74)


def panel_histograms(truth, predictions):
    true_p = np.linalg.norm(truth, axis=1)
    assert (true_p > 0).all() and true_p.max() <= P_EDGES[-1]
    method_residuals = []
    for prediction in predictions:
        _, dp, dtheta, dphi = errors(truth, prediction)
        method_residuals.append((100 * dp, np.degrees(dtheta), np.degrees(dphi)))

    panels = []
    for (_, _, limit, _), residuals in zip(PANELS, zip(*method_residuals)):
        y_edges = np.linspace(-limit, limit, 141)
        histograms = []
        for values in residuals:
            assert np.isfinite(values).all()
            counts, _, _ = np.histogram2d(true_p, values, bins=(P_EDGES, y_edges))
            outside = int(np.count_nonzero(np.abs(values) > limit))
            assert int(counts.sum()) + outside == len(true_p)
            histograms.append((counts.T, outside))
        panels.append((y_edges, histograms))
    return panels


def plot_row(fig, axes, panel, spec, *, show_headings):
    y_edges, histograms = panel
    _, ylabel, limit, major = spec
    norm = LogNorm(vmin=1, vmax=max(2, *(counts.max() for counts, _ in histograms)))
    for i, (ax, (counts, outside), method) in enumerate(zip(axes, histograms, METHODS)):
        image = ax.pcolormesh(
            P_EDGES, y_edges, np.ma.masked_equal(counts, 0),
            norm=norm, cmap="viridis", shading="flat", rasterized=True,
        )
        ax.axhline(0, color="#91a0ad", ls=(0, (3, 3)), lw=.7, alpha=.7)
        ax.set(xlim=(P_EDGES[0], P_EDGES[-1]), ylim=(-limit, limit))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(major))
        ax.tick_params(axis="both", length=3)
        ax.set_facecolor("#f6f8fa")
        if show_headings:
            ax.set_title(method, loc="left", fontsize=12, fontweight="bold", pad=22)
        ax.text(0, 1.018, f"{outside:,} outside plotted y range",
                transform=ax.transAxes, fontsize=8.5, color="#455466")
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(image, ax=axes, pad=.025, fraction=.04, label="Tracks per bin")


def save_plots(output, panels, tracks):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": "#202b36", "axes.labelcolor": "#202b36",
        "axes.edgecolor": "#a6b0ba", "axes.spines.right": False,
        "axes.spines.top": False, "xtick.color": "#455466", "ytick.color": "#455466",
        "savefig.facecolor": "white",
    })
    x_label = r"True momentum $p$ (GeV/$c$)"
    subtitle = f"{tracks:,} matched test tracks · Signed residuals relative to truth"
    fig, grid = plt.subplots(3, 2, figsize=(10.6, 9.3))
    fig.subplots_adjust(left=.09, right=.965, bottom=.075, top=.81, wspace=.075, hspace=.31)
    fig.suptitle("Momentum and angular residuals vs true momentum",
                 x=.09, y=.977, ha="left", fontsize=18, fontweight="bold")
    fig.text(.09, .927, subtitle, fontsize=10.5, color="#455466")
    for i, (row, panel, spec) in enumerate(zip(grid, panels, PANELS)):
        plot_row(fig, row, panel, spec, show_headings=i == 0)
        for ax in row:
            ax.tick_params(labelbottom=i == 2)
            if i == 2:
                ax.set_xlabel(x_label)
    fig.savefig(output / "residuals_vs_true_p.png", dpi=180)
    plt.close(fig)

    for panel, spec in zip(panels, PANELS):
        name, ylabel, _, _ = spec
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
        fig.subplots_adjust(left=.09, right=.965, bottom=.15, top=.63, wspace=.075)
        fig.suptitle(f"{ylabel} vs true momentum",
                     x=.09, y=.955, ha="left", fontsize=17, fontweight="bold")
        fig.text(.09, .857, subtitle, fontsize=10, color="#455466")
        plot_row(fig, axes, panel, spec, show_headings=True)
        for ax in axes:
            ax.set_xlabel(x_label)
        fig.savefig(output / f"{name}_vs_true_p.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((args.evidence / "selection.json").read_text())
    report = json.loads((args.evidence / "test_metrics.json").read_text())
    winner = selection["best_single"]["name"]
    assert winner == report["best_single_entry"]
    with np.load(args.evidence / "test_predictions.npz") as data:
        coatjava = data["coatjava"]
        paired = np.isfinite(coatjava).all(1)
        assert int(paired.sum()) == report["coatjava_paired"]["tracks"]
        truth = data["truth"][paired]
        predictions = (data[winner][paired], coatjava[paired])
    panels = panel_histograms(truth, predictions)
    save_plots(args.output, panels, len(truth))
    for (name, _, limit, _), (_, histograms) in zip(PANELS, panels):
        print(f"{name} ±{limit}: outside y range "
              + ", ".join(f"{method} {outside:,}" for method, (_, outside) in zip(METHODS, histograms)))


if __name__ == "__main__":
    main()
