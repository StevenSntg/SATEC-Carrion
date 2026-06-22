import os
import pandas as pd
from satec.models.figures import plot_metrics_bar
from satec.models.paper_style import nice_model


def test_plot_metrics_bar_crea_png(tmp_path):
    res = pd.DataFrame({"modelo": ["a", "b"], "recall": [0.5, 0.8],
                        "pr_auc": [0.4, 0.7], "f1": [0.45, 0.75]})
    out = tmp_path / "fig.png"
    plot_metrics_bar(res, str(out))
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_nombres_de_ensembles():
    assert nice_model("random_forest") == "Random Forest"
    assert nice_model("gradient_boosting") == "Gradient Boosting"
