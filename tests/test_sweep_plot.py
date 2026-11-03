import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional plotting dependency is not installed")
class SweepPlotTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.addCleanup(plt.close, "all")
        spec = importlib.util.spec_from_file_location(
            "sweep_plot", Path(__file__).resolve().parents[1] / "scripts/plot_label_sweep.py")
        self.plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.plot)

    def render(self, different_baseline=False):
        keys = ("ari_signal", "track_purity_global", "track_efficiency_global")
        baseline = {**dict.fromkeys(keys, .9), "n_events": 20}
        arm = {"summary": {"metrics": dict.fromkeys(keys, .8), "coatjava": baseline},
               "bootstrap": {"intervals_95": dict.fromkeys(keys, [.7, .9])}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {"arms": {"100": arm, "1000": arm}}
            (root / "egg.json").write_text(json.dumps(report))
            if different_baseline:
                baseline["n_events"] = 30
            (root / "exa.json").write_text(json.dumps(report))
            argv = ["plot", "--eggnet", str(root / "egg.json"), "--exatrkx", str(root / "exa.json"),
                    "--output", str(root / "plot.png")]
            with patch("sys.argv", argv):
                self.plot.main()
            self.assertGreater((root / "plot.png").stat().st_size, 0)

    def test_subset_of_budgets_renders_without_recipe_specific_claims(self):
        self.render()
        fig = self.plt.gcf()
        self.assertEqual([tick.get_text() for tick in fig.axes[0].get_xticklabels()], ["100", "1k"])
        self.assertFalse(any("100k updates" in text.get_text() or "5k validation" in text.get_text() for text in fig.texts))

    def test_different_evaluation_baselines_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "COATJAVA differs"):
            self.render(different_baseline=True)


if __name__ == "__main__":
    unittest.main()
