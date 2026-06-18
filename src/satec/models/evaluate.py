"""Orquestador de la evaluacion temporal de todos los modelos."""
import os
import numpy as np
import pandas as pd

from satec.models.features_matrix import feature_matrix
from satec.models.split import temporal_split
from satec.models.baselines import baseline_persistence
from satec.models.metrics import evaluate_predictions
from satec.models.train import (train_decision_tree, train_random_forest,
                                train_hist_gb, train_neural_net,
                                nn_predict_proba)


def _row(modelo, y_test, y_pred, y_score, train_acc):
    m = evaluate_predictions(y_test, y_pred, y_score)
    test_acc = m["accuracy"]
    gap = (train_acc - test_acc) if train_acc == train_acc else float("nan")
    m.update({"modelo": modelo, "train_acc": train_acc, "gap": gap})
    return m


def run_evaluation(df, year_cutoff=2019, nn_epochs=60) -> pd.DataFrame:
    train_df, test_df = temporal_split(df, year_cutoff)
    Xtr, ytr = feature_matrix(train_df)
    Xte, yte = feature_matrix(test_df)
    rows = []

    sklearn_models = {
        "arbol_sin_poda": train_decision_tree(Xtr, ytr, max_depth=None),
        "arbol_poda8": train_decision_tree(Xtr, ytr, max_depth=8),
        "random_forest": train_random_forest(Xtr, ytr),
        "hist_gb": train_hist_gb(Xtr, ytr),
    }
    for nombre, clf in sklearn_models.items():
        score = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        train_acc = float((clf.predict(Xtr) == ytr.to_numpy()).mean())
        rows.append(_row(nombre, yte, pred, score, train_acc))

    model, norm = train_neural_net(Xtr, ytr, epochs=nn_epochs)
    score = nn_predict_proba(model, Xte, norm)
    pred = (score >= 0.5).astype(int)
    train_score = nn_predict_proba(model, Xtr, norm)
    train_acc = float(((train_score >= 0.5).astype(int) == ytr.to_numpy()).mean())
    rows.append(_row("red_neuronal", yte, pred, score, train_acc))

    bp = baseline_persistence(test_df)
    rows.append(_row("baseline_persistencia", yte, bp, None, float("nan")))

    cols = ["modelo", "accuracy", "precision", "recall", "f1", "roc_auc",
            "pr_auc", "train_acc", "gap", "tn", "fp", "fn", "tp"]
    return pd.DataFrame(rows)[cols]


def main(repo: str = ".") -> None:
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    res = run_evaluation(df, year_cutoff=2019)
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    out = os.path.join(repo, "results", "metricas_modelos.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 160)
    print(res.to_string(index=False))
    print(f"\n[OK] -> {out}")


if __name__ == "__main__":
    main()
