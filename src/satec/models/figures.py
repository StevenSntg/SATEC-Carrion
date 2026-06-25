"""Figuras de resultados del modelado (calidad de revista)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from satec.models.paper_style import (apply_style, clean_axes, nice_model,
                                      PALETA, BERMELLON)

ORDEN = ["red_neuronal", "arbol_poda8", "arbol_sin_poda",
         "baseline_persistencia"]


def plot_metrics_bar(res_df, out_path):
    apply_style()
    metricas = ["recall", "pr_auc", "f1"]
    etiquetas = ["Sensibilidad", "AUC-PR", "F1"]
    sub = res_df.set_index("modelo").reindex([m for m in ORDEN
                                              if m in set(res_df["modelo"])])
    x = np.arange(len(sub)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, (m, lb) in enumerate(zip(metricas, etiquetas)):
        bars = ax.bar(x + (i - 1) * w, sub[m].to_numpy(dtype=float), w,
                      label=lb, color=PALETA[i], edgecolor="white", linewidth=0.5)
        for rect, val in zip(bars, sub[m].to_numpy(dtype=float)):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([nice_model(m) for m in sub.index])
    ax.set_ylabel("Valor de la métrica")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_generalization_gap(res_df, out_path):
    apply_style()
    sub = res_df.dropna(subset=["gap"]).set_index("modelo")
    sub = sub.reindex([m for m in ORDEN if m in sub.index])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    vals = sub["gap"].to_numpy(dtype=float)
    ax.bar([nice_model(m) for m in sub.index], vals, 0.55,
           color=BERMELLON, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_ylabel("Brecha de generalización\n(exactitud entren. − prueba)")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.0008 if v >= 0 else -0.0008), f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(repo: str = ".") -> None:
    import pandas as pd
    res = pd.read_csv(os.path.join(repo, "results", "metricas_modelos.csv"))
    plot_metrics_bar(res, os.path.join(repo, "results", "fig_metricas.png"))
    plot_generalization_gap(res, os.path.join(repo, "results", "fig_brecha.png"))
    print("[OK] figuras en results/")


if __name__ == "__main__":
    main()
