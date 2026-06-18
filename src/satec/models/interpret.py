"""Interpretabilidad: importancia por permutacion y calibracion de probabilidades."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.calibration import calibration_curve


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
    sub = imp_df.head(top).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(sub["feature"], sub["importance"], color="#38bdc9")
    plt.xlabel("Caida en AUC-PR al permutar la variable")
    plt.title("Importancia de variables (permutacion) — Red Neuronal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration(curves, out_path):
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "--", color="#92a4c4", label="Calibracion perfecta")
    for nombre, (pp, pt) in curves.items():
        plt.plot(pp, pt, "o-", label=nombre)
    plt.xlabel("Probabilidad predicha")
    plt.ylabel("Frecuencia observada de brote")
    plt.title("Curva de calibracion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
