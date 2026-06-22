"""Validación de origen móvil: entrena por corte temporal, elige umbral en
validación (sin tocar el año de prueba) y agrupa las predicciones fuera de
muestra de todos los años de prueba."""
import numpy as np
import pandas as pd

from satec.models.features_matrix import feature_matrix
from satec.models.metrics import evaluate_predictions
from satec.models.thresholds import best_threshold
from satec.models.train import (train_decision_tree, train_random_forest,
                                train_gradient_boosting, train_neural_net,
                                nn_predict_proba)


def _val_window(df, year, val_min_pos):
    """Años [year-1, year-2, ...] (estrictamente previos) hasta reunir
    val_min_pos brotes; devuelve el sub-DataFrame de validación."""
    val = df[df["anio"] == year - 1]
    k = 2
    while int(val["brote"].sum()) < val_min_pos and (year - k) >= df["anio"].min():
        val = df[(df["anio"] >= year - k) & (df["anio"] <= year - 1)]
        k += 1
    return val


def pooled_predictions(df, build_fn, score_fn, test_years,
                       val_min_pos=5, beta=1.0):
    yt, ys, yp, umbrales = [], [], [], []
    for year in test_years:
        train_inner = df[df["anio"] <= year - 2]
        val = _val_window(df, year, val_min_pos)
        test = df[df["anio"] == year]
        if len(train_inner) == 0 or len(test) == 0:
            continue
        Xtr, ytr = feature_matrix(train_inner)
        model = build_fn(Xtr, ytr)
        # umbral en validación; si no hay validación útil, 0.5
        if len(val) and int(val["brote"].sum()) > 0:
            Xv, yv = feature_matrix(val)
            t, _ = best_threshold(yv, score_fn(model, Xv), beta=beta)
        else:
            t = 0.5
        Xte, yte = feature_matrix(test)
        s = score_fn(model, Xte)
        yt.append(yte.to_numpy()); ys.append(s)
        yp.append((s >= t).astype(int)); umbrales.append((int(year), float(t)))
    return {"y_true": np.concatenate(yt), "y_score": np.concatenate(ys),
            "y_pred": np.concatenate(yp), "umbrales": umbrales}


def _row(modelo, pooled):
    m = evaluate_predictions(pooled["y_true"], pooled["y_pred"],
                             pooled["y_score"])
    ts = [t for _, t in pooled["umbrales"]]
    m.update({"modelo": modelo,
              "umbral": float(np.mean(ts)) if ts else 0.5})
    return m


def _nn_build(Xtr, ytr):
    return train_neural_net(Xtr, ytr, epochs=_nn_build.epochs)


def rolling_evaluation(df, test_years=range(2016, 2025), nn_epochs=60, beta=1.0):
    test_years = list(test_years)
    rows = []

    especificaciones = {
        "arbol_sin_poda": (lambda X, y: train_decision_tree(X, y, max_depth=None),
                           lambda m, X: m.predict_proba(X)[:, 1]),
        "arbol_poda8": (lambda X, y: train_decision_tree(X, y, max_depth=8),
                        lambda m, X: m.predict_proba(X)[:, 1]),
        "random_forest": (lambda X, y: train_random_forest(X, y),
                          lambda m, X: m.predict_proba(X)[:, 1]),
        "gradient_boosting": (lambda X, y: train_gradient_boosting(X, y),
                              lambda m, X: m.predict_proba(X)[:, 1]),
    }
    for nombre, (build, score) in especificaciones.items():
        pooled = pooled_predictions(df, build, score, test_years, beta=beta)
        rows.append(_row(nombre, pooled))

    # Red neuronal: build devuelve (model, norm); score usa nn_predict_proba.
    _nn_build.epochs = nn_epochs
    pooled_rn = pooled_predictions(
        df, _nn_build,
        lambda mn, X: nn_predict_proba(mn[0], X, mn[1]),
        test_years, beta=beta)
    rows.append(_row("red_neuronal", pooled_rn))

    # Baseline canal endémico: marca brote si casos > q3 (umbral fijo, sin score).
    yt, yp = [], []
    for year in test_years:
        test = df[df["anio"] == year]
        if len(test) == 0:
            continue
        _, yte = feature_matrix(test)
        yt.append(yte.to_numpy())
        yp.append((test["casos"] > test["q3"]).astype(int).to_numpy())
    bp = {"y_true": np.concatenate(yt), "y_pred": np.concatenate(yp),
          "y_score": None, "umbrales": []}
    m = evaluate_predictions(bp["y_true"], bp["y_pred"])
    m.update({"modelo": "baseline_persistencia", "umbral": float("nan")})
    rows.append(m)

    cols = ["modelo", "accuracy", "precision", "recall", "f1", "f2",
            "especificidad", "roc_auc", "pr_auc", "brier", "umbral",
            "tn", "fp", "fn", "tp"]
    return pd.DataFrame(rows)[cols]
