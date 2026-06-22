import numpy as np
from satec.models.thresholds import best_threshold


def test_best_threshold_separa_clases():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    t, f = best_threshold(y_true, y_score, beta=1.0)
    assert 0.2 < t <= 0.8     # cualquier umbral en ese rango da F1=1
    assert abs(f - 1.0) < 1e-9


def test_best_threshold_sin_positivos_devuelve_default():
    t, f = best_threshold([0, 0, 0], [0.1, 0.2, 0.3])
    assert t == 0.5 and f == 0.0
