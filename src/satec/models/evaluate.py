"""Orquestador de la evaluacion temporal de todos los modelos."""
import os
import numpy as np
import pandas as pd

from satec.models.features_matrix import feature_matrix
from satec.models.split import temporal_split
from satec.models.baselines import baseline_persistence
from satec.models.metrics import evaluate_predictions
from satec.models.train import (train_decision_tree, train_random_forest,
                                train_gradient_boosting, train_neural_net,
                                nn_predict_proba)
from satec.models.thresholds import best_threshold
from satec.models.rolling_origin import rolling_evaluation


def _row(modelo, y_test, y_pred, y_score, train_acc):
    m = evaluate_predictions(y_test, y_pred, y_score)
    test_acc = m["accuracy"]
    gap = (train_acc - test_acc) if train_acc == train_acc else float("nan")
    m.update({"modelo": modelo, "train_acc": train_acc, "gap": gap})
    m.setdefault("umbral", 0.5)
    return m


def run_evaluation(df, year_cutoff=2019, nn_epochs=60) -> pd.DataFrame:
    train_df, test_df = temporal_split(df, year_cutoff)
    Xtr, ytr = feature_matrix(train_df)
    Xte, yte = feature_matrix(test_df)
    rows = []

    arboles = {
        "arbol_sin_poda": train_decision_tree(Xtr, ytr, max_depth=None),
        "arbol_poda8": train_decision_tree(Xtr, ytr, max_depth=8),
    }
    for nombre, clf in arboles.items():
        score = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        train_acc = float((clf.predict(Xtr) == ytr.to_numpy()).mean())
        rows.append(_row(nombre, yte, pred, score, train_acc))

    ensembles = {
        "random_forest": train_random_forest(Xtr, ytr),
        "gradient_boosting": train_gradient_boosting(Xtr, ytr),
    }
    for nombre, clf in ensembles.items():
        score = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        train_acc = float((clf.predict(Xtr) == ytr.to_numpy()).mean())
        rows.append(_row(nombre, yte, pred, score, train_acc))

    model, norm = train_neural_net(Xtr, ytr, epochs=nn_epochs)
    score = nn_predict_proba(model, Xte, norm)
    val_year = int(train_df["anio"].max())
    val = train_df[train_df["anio"] == val_year]
    if len(val) and int(val["brote"].sum()) > 0:
        Xv, yv = feature_matrix(val)
        t_rn, _ = best_threshold(yv, nn_predict_proba(model, Xv, norm))
    else:
        t_rn = 0.5
    pred = (score >= t_rn).astype(int)
    train_score = nn_predict_proba(model, Xtr, norm)
    train_acc = float(((train_score >= t_rn).astype(int) == ytr.to_numpy()).mean())
    rows.append(_row("red_neuronal", yte, pred, score, train_acc))
    rows[-1]["umbral"] = t_rn

    bp = baseline_persistence(test_df)
    rows.append(_row("baseline_persistencia", yte, bp, None, float("nan")))

    cols = ["modelo", "accuracy", "precision", "recall", "f1", "f2",
            "especificidad", "roc_auc", "pr_auc", "brier",
            "umbral", "train_acc", "gap", "tn", "fp", "fn", "tp"]
    return pd.DataFrame(rows)[cols]


def main(repo: str = ".", modo: str = "rolling") -> None:
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    if modo == "rolling":
        res = rolling_evaluation(df, test_years=range(2016, 2025))
        out = os.path.join(repo, "results", "metricas_modelos.csv")
    else:
        res = run_evaluation(df, year_cutoff=2019)
        out = os.path.join(repo, "results", "metricas_modelos_corte_unico.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 180)
    print(res.to_string(index=False))
    print(f"\n[OK] -> {out}")


if __name__ == "__main__":
    main()
