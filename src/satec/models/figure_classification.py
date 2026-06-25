"""Figura tipo tabla con el classification_report (scikit-learn) de cada modelo.

Reproduce el formato estándar de ``sklearn.metrics.classification_report``
(precisión, sensibilidad, F1 y soporte por clase, más exactitud y promedios)
calculado con el umbral de decisión óptimo del artículo, de modo que las cifras
de la clase «Brote» coinciden con la Tabla 1."""
import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from satec.models.paper_style import apply_style, AZUL, nice_model, t

MODELOS = ["red_neuronal", "arbol_poda8", "arbol_sin_poda"]


def _miles(n):
    return f"{int(n):,}".replace(",", " ")  # separador fino de miles


def _rows(tn, fp, fn, tp):
    y_true = [0] * (tn + fp) + [1] * (fn + tp)
    y_pred = [0] * tn + [1] * fp + [0] * fn + [1] * tp
    d = classification_report(y_true, y_pred, target_names=["neg", "pos"],
                              output_dict=True, zero_division=0)

    def f(b):
        return [f"{b['precision']:.2f}", f"{b['recall']:.2f}",
                f"{b['f1-score']:.2f}", _miles(b["support"])]
    tot = _miles(d["macro avg"]["support"])
    return [
        f(d["neg"]),
        f(d["pos"]),
        ["", "", f"{d['accuracy']:.2f}", tot],
        f(d["macro avg"]),
        f(d["weighted avg"]),
    ]


def plot_classification(out_path, repo: str = ".", lang="es"):
    apply_style()
    df = pd.read_csv(os.path.join(repo, "results", "metricas_modelos.csv")
                     ).set_index("modelo")
    cols = [t("precision", lang), t("recall", lang), t("f1", lang), t("support", lang)]
    filas = [t("no_outbreak", lang), t("outbreak", lang), t("accuracy", lang),
             t("macro_avg", lang), t("weighted_avg", lang)]
    fig, axes = plt.subplots(len(MODELOS), 1, figsize=(7.4, 7.0))
    for ax, k in zip(axes, MODELOS):
        r = df.loc[k]
        rows = _rows(int(r.tn), int(r.fp), int(r.fn), int(r.tp))
        tab = ax.table(cellText=rows, rowLabels=filas, colLabels=cols,
                       cellLoc="center", rowLoc="right", loc="center",
                       colWidths=[0.17, 0.20, 0.13, 0.15])
        tab.auto_set_font_size(False)
        tab.set_fontsize(9.5)
        tab.scale(1, 1.55)
        for (i, j), cell in tab.get_celld().items():
            cell.set_edgecolor("#FFFFFF")
            cell.set_linewidth(1.0)
            if i == 0:  # cabecera de columnas
                cell.set_facecolor("#E9EEF2")
                cell.set_text_props(fontweight="bold", color="#1A1A1A")
            elif j == -1:  # etiquetas de fila
                cell.set_text_props(ha="right", color="#333333")
                cell.set_facecolor("#F6F8FA")
            elif i == 2:  # fila «Brote» (clase de interés) resaltada
                cell.set_facecolor(AZUL + "26")
                cell.set_text_props(fontweight="bold")
            elif i in (3, 5, 6):  # separación visual de exactitud/promedios
                cell.set_facecolor("#FBFCFD")
            else:
                cell.set_facecolor("#FFFFFF")
        tab[2, -1].set_facecolor(AZUL + "26")
        tab[2, -1].set_text_props(fontweight="bold", ha="right", color="#1A1A1A")
        ax.set_title(f"{nice_model(k, lang)}  ·  {t('test', lang)} 2020–2024  ·  "
                     f"{t('threshold', lang)} {r.umbral:.2f}",
                     loc="left", fontsize=10.5, fontweight="bold", pad=10)
        ax.axis("off")
    fig.tight_layout(h_pad=2.2)
    fig.savefig(out_path)
    plt.close(fig)


def main(repo: str = ".", lang: str = "es", out=None) -> None:
    out = out or os.path.join(repo, "results", "fig_classification.png")
    plot_classification(out, repo, lang)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
