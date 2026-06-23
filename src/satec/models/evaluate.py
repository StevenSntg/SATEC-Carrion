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


def _row(modelo, y_test, y_pred, y_score, train_acc, umbral):
    m = evaluate_predictions(y_test, y_pred, y_score)
    test_acc = m["accuracy"]
    gap = (train_acc - test_acc) if train_acc == train_acc else float("nan")
    m.update({"modelo": modelo, "train_acc": train_acc, "gap": gap,
              "umbral": umbral})
    return m


def run_evaluation(df, year_cutoff=2019, nn_epochs=60) -> pd.DataFrame:
    """Validacion temporal de corte unico: entrena con anios < year_cutoff,
    elige el umbral en el anio de validacion (= year_cutoff) y prueba con los
    anios > year_cutoff. Sin fuga: el conjunto de prueba no interviene ni en el
    ajuste ni en la seleccion del umbral. Modelos: arbol sin poda, Gradient
    Boosting y red neuronal, mas el baseline del canal endemico."""
    train_df = df[df["anio"] < year_cutoff].reset_index(drop=True)
    val_df = df[df["anio"] == year_cutoff].reset_index(drop=True)
    test_df = df[df["anio"] > year_cutoff].reset_index(drop=True)
    Xtr, ytr = feature_matrix(train_df)
    Xv, yv = feature_matrix(val_df)
    Xte, yte = feature_matrix(test_df)
    rows = []

    def add(nombre, s_val, s_tr, s_te):
        if int(yv.sum()) > 0:
            t, _ = best_threshold(yv.to_numpy(), s_val)
        else:
            t = 0.5
        pred = (s_te >= t).astype(int)
        train_acc = float(((s_tr >= t).astype(int) == ytr.to_numpy()).mean())
        rows.append(_row(nombre, yte, pred, s_te, train_acc, float(t)))

    ad = train_decision_tree(Xtr, ytr, max_depth=None)
    add("arbol_sin_poda", ad.predict_proba(Xv)[:, 1],
        ad.predict_proba(Xtr)[:, 1], ad.predict_proba(Xte)[:, 1])

    gb = train_gradient_boosting(Xtr, ytr)
    add("gradient_boosting", gb.predict_proba(Xv)[:, 1],
        gb.predict_proba(Xtr)[:, 1], gb.predict_proba(Xte)[:, 1])

    model, norm = train_neural_net(Xtr, ytr, epochs=nn_epochs)
    add("red_neuronal", nn_predict_proba(model, Xv, norm),
        nn_predict_proba(model, Xtr, norm), nn_predict_proba(model, Xte, norm))

    bp = baseline_persistence(test_df)
    m = evaluate_predictions(yte, bp)
    m.update({"modelo": "baseline_persistencia", "train_acc": float("nan"),
              "gap": float("nan"), "umbral": float("nan")})
    rows.append(m)

    cols = ["modelo", "accuracy", "precision", "recall", "f1", "f2",
            "especificidad", "roc_auc", "pr_auc", "brier",
            "umbral", "train_acc", "gap", "tn", "fp", "fn", "tp"]
    return pd.DataFrame(rows)[cols]


def main(repo: str = ".", modo: str = "corte_unico") -> None:
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    if modo == "rolling":
        res = rolling_evaluation(df, test_years=range(2016, 2025))
        out = os.path.join(repo, "results", "metricas_modelos_robustez.csv")
    else:
        res = run_evaluation(df, year_cutoff=2019)
        out = os.path.join(repo, "results", "metricas_modelos.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 180)
    print(res.to_string(index=False))
    print(f"\n[OK] -> {out}")


if __name__ == "__main__":
    main()
