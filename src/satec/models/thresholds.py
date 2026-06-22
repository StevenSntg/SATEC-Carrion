"""Selección de umbral de decisión que maximiza F-beta (sin tocar el test)."""
import numpy as np
from sklearn.metrics import fbeta_score


def best_threshold(y_true, y_score, beta=1.0):
    """Devuelve (t*, f*): el umbral que maximiza F-beta sobre y_score.

    Si no hay positivos en y_true no hay señal para optimizar y se devuelve el
    umbral neutro 0.5.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0:
        return 0.5, 0.0
    candidatos = np.unique(np.concatenate([[0.0], np.sort(y_score), [1.0]]))
    best_t, best_f = 0.5, -1.0
    for t in candidatos:
        pred = (y_score >= t).astype(int)
        f = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, float(best_f)
