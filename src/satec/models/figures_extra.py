"""Figuras extra del paper: matrices de confusion y metricas derivadas (calidad de revista)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from satec.models.paper_style import apply_style, clean_axes, nice_model, t, PALETA

# Mapa de color sobrio (blanco -> azul ACM) para las matrices de confusion.
_CMAP = LinearSegmentedColormap.from_list("acmblue", ["#FFFFFF", "#0072B2"])


def plot_confusions(res, out_path, lang="es"):
    apply_style()
    modelos = ["red_neuronal", "arbol_poda8"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))
    for ax, m in zip(axes, modelos):
        r = res[res["modelo"] == m].iloc[0]
        cm = np.array([[int(r["tn"]), int(r["fp"])], [int(r["fn"]), int(r["tp"])]])
        norm = cm / cm.max()
        ax.imshow(norm, cmap=_CMAP, vmin=0, vmax=1)
        total = cm.sum()
        etiquetas = [[t("TN", lang), t("FP", lang)], [t("FN", lang), t("TP", lang)]]
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / total
                ax.text(j, i, f"{etiquetas[i][j]}\n{cm[i, j]:,}\n{pct:.2f}%",
                        ha="center", va="center",
                        color="white" if norm[i, j] > 0.55 else "#222222",
                        fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels([t("no_outbreak", lang), t("outbreak", lang)])
        ax.set_yticks([0, 1]); ax.set_yticklabels([t("no_outbreak", lang), t("outbreak", lang)])
        ax.set_xlabel(t("prediction", lang)); ax.set_ylabel(t("observed", lang))
        ax.set_title(nice_model(m, lang), pad=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_f1(res, out_path):
    apply_style()
    res = res.copy()
    res["especificidad"] = res["tn"] / (res["tn"] + res["fp"])
    modelos = ["red_neuronal", "arbol_poda8", "arbol_sin_poda",
               "baseline_persistencia"]
    modelos = [m for m in modelos if m in set(res["modelo"])]
    sub = res[res["modelo"].isin(modelos)].set_index("modelo").loc[modelos]
    mets = ["precision", "recall", "f1", "especificidad"]
    labels = ["Precisión", "Sensibilidad", "F1", "Especificidad"]
    x = np.arange(len(modelos)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (mt, lb) in enumerate(zip(mets, labels)):
        bars = ax.bar(x + (i - 1.5) * w, sub[mt].to_numpy(dtype=float), w,
                      label=lb, color=PALETA[i], edgecolor="white", linewidth=0.5)
        for rect, val in zip(bars, sub[mt].to_numpy(dtype=float)):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([nice_model(m) for m in modelos])
    ax.set_ylabel("Valor de la métrica"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.13))
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(repo: str = ".") -> None:
    res = pd.read_csv(os.path.join(repo, "results", "metricas_modelos.csv"))
    plot_confusions(res, os.path.join(repo, "results", "fig_confusion.png"))
    plot_f1(res, os.path.join(repo, "results", "fig_f1.png"))
    print("[OK] fig_confusion.png y fig_f1.png en results/")


if __name__ == "__main__":
    main()
