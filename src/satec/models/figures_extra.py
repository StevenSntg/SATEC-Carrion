"""Figuras extra del paper: matrices de confusion y metricas derivadas (F1, etc.)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOMBRES = {
    "red_neuronal": "Red Neuronal",
    "arbol_poda8": "Arbol (poda 8)",
    "arbol_sin_poda": "Arbol (sin poda)",
    "baseline_persistencia": "Canal endemico",
}


def plot_confusions(res, out_path):
    modelos = ["red_neuronal", "arbol_poda8"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, m in zip(axes, modelos):
        r = res[res["modelo"] == m].iloc[0]
        cm = np.array([[int(r["tn"]), int(r["fp"])], [int(r["fn"]), int(r["tp"])]])
        ax.imshow(cm, cmap="cividis")
        thr = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], ","), ha="center", va="center",
                        color="white" if cm[i, j] < thr else "black", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No brote", "Brote"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No brote", "Brote"])
        ax.set_xlabel("Prediccion"); ax.set_ylabel("Observado")
        ax.set_title(NOMBRES[m])
    fig.suptitle("Matrices de confusion (conjunto de prueba 2020-2024)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_f1(res, out_path):
    res = res.copy()
    res["especificidad"] = res["tn"] / (res["tn"] + res["fp"])
    modelos = ["red_neuronal", "arbol_poda8", "arbol_sin_poda", "baseline_persistencia"]
    sub = res[res["modelo"].isin(modelos)].set_index("modelo").loc[modelos]
    mets = ["precision", "recall", "f1", "especificidad"]
    labels = ["Precision", "Sensibilidad", "F1", "Especificidad"]
    x = np.arange(len(modelos)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (mt, lb) in enumerate(zip(mets, labels)):
        ax.bar(x + (i - 1.5) * w, sub[mt].to_numpy(), w, label=lb)
    ax.set_xticks(x)
    ax.set_xticklabels([NOMBRES[m] for m in modelos], rotation=12, ha="right")
    ax.set_ylabel("Valor"); ax.set_ylim(0, 1.05)
    ax.set_title("Metricas derivadas de la matriz de confusion por modelo")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(repo: str = ".") -> None:
    res = pd.read_csv(os.path.join(repo, "results", "metricas_modelos.csv"))
    plot_confusions(res, os.path.join(repo, "results", "fig_confusion.png"))
    plot_f1(res, os.path.join(repo, "results", "fig_f1.png"))
    print("[OK] fig_confusion.png y fig_f1.png en results/")


if __name__ == "__main__":
    main()
