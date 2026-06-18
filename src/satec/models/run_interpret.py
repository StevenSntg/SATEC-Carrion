"""Genera figuras de interpretabilidad: importancia (permutacion) y calibracion."""
import os
import pandas as pd

from satec.models.features_matrix import feature_matrix
from satec.models.split import temporal_split
from satec.models.train import (train_decision_tree, train_neural_net,
                                nn_predict_proba)
from satec.models.interpret import (permutation_importance_ap,
                                    calibration_points, plot_importance,
                                    plot_calibration)


def main(repo: str = ".", year_cutoff: int = 2019) -> None:
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    train_df, test_df = temporal_split(df, year_cutoff)
    Xtr, ytr = feature_matrix(train_df)
    Xte, yte = feature_matrix(test_df)

    ad = train_decision_tree(Xtr, ytr, max_depth=8)
    model, norm = train_neural_net(Xtr, ytr, epochs=60)

    def rn_pp(X):
        return nn_predict_proba(model, X, norm)

    results = os.path.join(repo, "results")
    os.makedirs(results, exist_ok=True)

    imp = permutation_importance_ap(rn_pp, Xte, yte, n_repeats=5)
    imp.to_csv(os.path.join(results, "importancia_variables.csv"), index=False)
    plot_importance(imp, os.path.join(results, "fig_importancia.png"))

    curves = {
        "Red Neuronal": calibration_points(yte, rn_pp(Xte), n_bins=8),
        "Arbol (poda 8)": calibration_points(
            yte, ad.predict_proba(Xte)[:, 1], n_bins=8),
    }
    plot_calibration(curves, os.path.join(results, "fig_calibracion.png"))
    print("[OK] fig_importancia.png · fig_calibracion.png · importancia_variables.csv")
    print(imp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
