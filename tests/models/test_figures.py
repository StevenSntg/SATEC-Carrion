import os
import pandas as pd
from satec.models.figures import plot_metrics_bar


def test_plot_metrics_bar_crea_png(tmp_path):
    res = pd.DataFrame({"modelo": ["a", "b"], "recall": [0.5, 0.8],
                        "pr_auc": [0.4, 0.7], "f1": [0.45, 0.75]})
    out = tmp_path / "fig.png"
    plot_metrics_bar(res, str(out))
    assert os.path.exists(out) and os.path.getsize(out) > 0
