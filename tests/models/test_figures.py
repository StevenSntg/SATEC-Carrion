import os
import pandas as pd
from satec.models.figures import plot_metrics_bar
from satec.models.paper_style import nice_model
from satec.models.figures_extra import plot_confusions


def test_plot_metrics_bar_crea_png(tmp_path):
    res = pd.DataFrame({"modelo": ["a", "b"], "recall": [0.5, 0.8],
                        "pr_auc": [0.4, 0.7], "f1": [0.45, 0.75]})
    out = tmp_path / "fig.png"
    plot_metrics_bar(res, str(out))
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_nombres_de_modelos():
    assert nice_model("red_neuronal") == "Red neuronal"
    assert nice_model("arbol_poda8") == "Árbol (poda 8)"
    assert nice_model("arbol_sin_poda") == "Árbol (sin poda)"


def test_metrics_bar_con_modelos(tmp_path):
    filas = []
    for m in ["red_neuronal", "arbol_poda8", "arbol_sin_poda",
              "baseline_persistencia"]:
        filas.append({"modelo": m, "recall": 0.5, "pr_auc": 0.5, "f1": 0.5})
    df = pd.DataFrame(filas)
    out = tmp_path / "m.png"
    plot_metrics_bar(df, str(out))
    assert out.exists()


def test_confusiones_rn_y_arbol(tmp_path):
    filas = []
    for m in ["red_neuronal", "arbol_poda8"]:
        filas.append({"modelo": m, "tn": 100, "fp": 10, "fn": 5, "tp": 20})
    df = pd.DataFrame(filas)
    out = tmp_path / "c.png"
    plot_confusions(df, str(out))
    assert out.exists()
