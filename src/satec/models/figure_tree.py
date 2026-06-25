"""Figura del árbol de decisión: primeros niveles del árbol podado (max_depth=8),
que es el modelo de árbol del artículo.

Se muestran sus tres primeros niveles —las particiones más informativas, encabezadas
por la media móvil de casos— como ilustración de las reglas legibles que aprende."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

from satec.models.features_matrix import feature_matrix, FEATURE_COLS
from satec.models.train import train_decision_tree
from satec.models.paper_style import apply_style, nice_var, t


def plot_decision_tree(out_path, repo: str = ".", max_depth_show: int = 3, lang="es"):
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    train = df[df["anio"] < 2019]
    X, y = feature_matrix(train)
    clf = train_decision_tree(X, y, max_depth=8)  # árbol podado (modelo del paper)

    apply_style()
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    plot_tree(clf, max_depth=max_depth_show, ax=ax,
              feature_names=[nice_var(c, lang) for c in FEATURE_COLS],
              class_names=[t("no_outbreak", lang), t("outbreak", lang)], filled=True, rounded=True,
              impurity=False, proportion=True, fontsize=8, precision=2)
    ax.set_title("")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return clf.get_depth(), clf.tree_.node_count


def main(repo: str = ".", lang: str = "es", out=None) -> None:
    out = out or os.path.join(repo, "results", "fig_arbol.png")
    d, n = plot_decision_tree(out, repo, lang=lang)
    print(f"[OK] {out} (profundidad {d}, {n} nodos)")


if __name__ == "__main__":
    main()
