"""Interpretabilidad: importancia por permutacion y calibracion de probabilidades."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.calibration import calibration_curve

from satec.models.paper_style import (apply_style, clean_axes, nice_var,
                                      AZUL, NARANJA, GRIS)


def permutation_importance_ap(predict_proba_fn, X, y, n_repeats=5, seed=42):
    """Importancia por permutacion = caida en AUC-PR al permutar cada variable.

    Generica: recibe una funcion predict_proba(X)->scores, asi sirve igual para
    los modelos de sklearn y para la red neuronal.
    """
    rng = np.random.RandomState(seed)
    base = average_precision_score(y, predict_proba_fn(X))
    filas = []
    for col in X.columns:
        caidas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            caidas.append(base - average_precision_score(y, predict_proba_fn(Xp)))
        filas.append({"feature": col, "importance": float(np.mean(caidas))})
    return (pd.DataFrame(filas).sort_values("importance", ascending=False)
              .reset_index(drop=True))


def calibration_points(y_true, y_score, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="quantile")
    return prob_pred, prob_true


def plot_importance(imp_df, out_path, top=12):
    apply_style()
    sub = imp_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    nombres = [nice_var(f) for f in sub["feature"]]
    ax.barh(nombres, sub["importance"].to_numpy(dtype=float),
            color=AZUL, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Caída en AUC-PR al permutar la variable")
    clean_axes(ax, grid_axis="x")
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_calibration(curves, out_path):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], "--", color=GRIS, linewidth=1.2,
            label="Calibración perfecta")
    colores = [AZUL, NARANJA]
    for (nombre, (pp, pt)), col in zip(curves.items(), colores):
        ax.plot(pp, pt, "o-", color=col, markersize=5, linewidth=1.6,
                label=nombre)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada de brote")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    clean_axes(ax)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
