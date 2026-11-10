"""Plot audited sweep summaries with conditional event-bootstrap intervals."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from clas12_benchmarks.finding_evaluation import require_same_population


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eggnet", required=True)
    p.add_argument("--exatrkx", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    reports = [("EggNet", "#2878B5", json.loads(Path(args.eggnet).read_text())),
               ("Exa.TrkX", "#D65A23", json.loads(Path(args.exatrkx).read_text()))]
    budgets_all = sorted({int(n) for _, _, report in reports for n in report["arms"]})
    if not budgets_all or min(budgets_all) < 1 or any(not report["arms"] for _, _, report in reports):
        raise ValueError("Both reports must contain positive training budgets")
    baseline = reports[0][2]["arms"][str(min(map(int, reports[0][2]["arms"])))]["summary"]["coatjava"]
    first = next(iter(reports[0][2]["arms"].values()))["summary"]
    for _, _, report in reports:
        for arm in report["arms"].values():
            require_same_population(first, arm["summary"])
    if any(arm["summary"]["coatjava"] != baseline
           for _, _, report in reports for arm in report["arms"].values()):
        raise ValueError("Evaluation population/COATJAVA differs between plotted arms")
    panels = [("ari_signal", "Signal ARI"),
              ("track_purity_global", "Track purity (global)"),
              ("track_efficiency_global", "Track efficiency (global)")]
    if "mc_track_dice_mean" in first["metrics"]:
        panels = [("mc_track_dice_mean", "MC track Dice"),
                  ("mc_hit_purity_global", "MC hit purity (global)"),
                  ("mc_hit_completeness_global", "MC hit completeness (global)"),
                  ("track_efficiency_global", "Accepted-track efficiency")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2*len(panels), 4.3), sharex=True)
    for label, color, report in reports:
        budgets = sorted(map(int, report["arms"]))
        for ax, (key, title) in zip(axes, panels):
            rows = [report["arms"][str(n)] for n in budgets]
            y = [r["summary"]["metrics"][key] if r["summary"]["metrics"][key] is not None else np.nan for r in rows]
            bounds = np.array([r["bootstrap"]["intervals_95"][key] or [np.nan, np.nan] for r in rows])
            ax.plot(budgets, y, "o-", label=label, color=color, linewidth=2)
            ax.fill_between(budgets, bounds[:, 0], bounds[:, 1], color=color, alpha=.18)
            ax.set_title(title)
            ax.set_xscale("log")
            ax.set_xticks(budgets_all, [f"{n // 1000}k" if n % 1000 == 0 else str(n) for n in budgets_all])
            ax.set_ylim(0, 1)
            ax.set_xlabel("Labeled training events")
            ax.grid(alpha=.2)
    for ax, (key, _) in zip(axes, panels):
        y = baseline[key]
        if y is not None:
            ax.axhline(y, linestyle="--", color="#555555", linewidth=1.3, label="Stored COATJAVA")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .98))
    fig.suptitle("CLAS12 training-label sweep", y=1.04, fontsize=15)
    fig.text(.5, .01, "Training budgets exclude validation labels. Bands: 95% event bootstrap; training-seed uncertainty excluded.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .88))
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
