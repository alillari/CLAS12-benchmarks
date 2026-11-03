"""Plot audited sweep summaries with conditional event-bootstrap intervals."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eggnet", required=True)
    p.add_argument("--exatrkx", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    reports = [("EggNet", "#2878B5", json.loads(Path(args.eggnet).read_text())),
               ("Exa.TrkX", "#D65A23", json.loads(Path(args.exatrkx).read_text()))]
    panels = [("ari_signal", "Signal ARI"),
              ("track_purity_global", "Track purity (global)"),
              ("track_efficiency_global", "Track efficiency (global)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharex=True)
    for label, color, report in reports:
        budgets = sorted(map(int, report["arms"]))
        for ax, (key, title) in zip(axes, panels):
            rows = [report["arms"][str(n)] for n in budgets]
            y = [r["summary"]["metrics"][key] for r in rows]
            bounds = np.array([r["bootstrap"]["intervals_95"][key] or [np.nan, np.nan] for r in rows])
            ax.plot(budgets, y, "o-", label=label, color=color, linewidth=2)
            ax.fill_between(budgets, bounds[:, 0], bounds[:, 1], color=color, alpha=.18)
            ax.set_title(title)
            ax.set_xscale("log")
            ax.set_xticks(budgets, ["100", "1k", "10k", "100k"])
            ax.set_ylim(0, 1)
            ax.set_xlabel("Labeled training events")
            ax.grid(alpha=.2)
    baseline = reports[0][2]["arms"][str(min(map(int, reports[0][2]["arms"])))]["summary"]["coatjava"]
    for ax, (key, _) in zip(axes, panels):
        y = baseline[key]
        ax.axhline(y, linestyle="--", color="#555555", linewidth=1.3, label="Stored COATJAVA")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .98))
    fig.suptitle("CLAS12 training-label sweep", y=1.04, fontsize=15)
    fig.text(.5, .01, "Fixed 100k updates/stage; 5k validation labels additional. Bands: 95% event bootstrap, one trained seed.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .88))
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
