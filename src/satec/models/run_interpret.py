"""Genera figuras de interpretabilidad: importancia (permutacion) y calibracion."""
import os
import pandas as pd
from satec.models.rolling_origin import pooled_predictions, _nn_build
from satec.models.train import (train_random_forest, train_neural_net,
                                nn_predict_proba)
from satec.models.features_matrix import feature_matrix
from satec.models.interpret import (permutation_importance_ap,
                                    calibration_points, plot_importance,
                                    plot_calibration)
from satec.models.paper_style import nice_model


def main(repo: str = ".") -> None:
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    results = os.path.join(repo, "results"); os.makedirs(results, exist_ok=True)
    years = range(2016, 2025)

    rf_pool = pooled_predictions(
        df, lambda X, y: train_random_forest(X, y),
        lambda m, X: m.predict_proba(X)[:, 1], years)
    _nn_build.epochs = 60
    rn_pool = pooled_predictions(
        df, _nn_build, lambda mn, X: nn_predict_proba(mn[0], X, mn[1]), years)

    curves = {
        nice_model("red_neuronal"): calibration_points(
            rn_pool["y_true"], rn_pool["y_score"], n_bins=10),
        nice_model("random_forest"): calibration_points(
            rf_pool["y_true"], rf_pool["y_score"], n_bins=10),
    }
    plot_calibration(curves, os.path.join(results, "fig_calibracion.png"))

    # Importancia por permutación de la RN entrenada con todo el histórico.
    Xall, yall = feature_matrix(df[df["anio"] <= 2024])
    model, norm = train_neural_net(Xall, yall, epochs=60)
    imp = permutation_importance_ap(
        lambda X: nn_predict_proba(model, X, norm), Xall, yall, n_repeats=5)
    imp.to_csv(os.path.join(results, "importancia_variables.csv"), index=False)
    plot_importance(imp, os.path.join(results, "fig_importancia.png"))
    print("[OK] fig_calibracion.png · fig_importancia.png")


if __name__ == "__main__":
    main()
