import numpy as np
import pandas as pd
from satec.models.features_matrix import FEATURE_COLS
from satec.models.evaluate import run_evaluation


def _dataset(n_per_year=200):
    rng = np.random.RandomState(0)
    rows = []
    for anio in range(2015, 2022):
        for _ in range(n_per_year):
            casos = rng.poisson(1.0)
            q3 = 1.0
            feats = {c: rng.rand() for c in FEATURE_COLS}
            feats["casos"] = float(casos)
            brote = int(casos > q3)
            rows.append({**feats, "departamento": "X", "provincia": "P",
                         "ubigeo_prov": "0201", "anio": anio, "semana": 5,
                         "q1": 0.0, "q2": 0.0, "q3": q3, "brote": brote})
    return pd.DataFrame(rows)


def test_run_evaluation_devuelve_metricas_por_modelo():
    res = run_evaluation(_dataset(), year_cutoff=2019, nn_epochs=3)
    modelos = set(res["modelo"])
    assert {"arbol_sin_poda", "red_neuronal", "random_forest",
            "hist_gb", "baseline_persistencia"} <= modelos
    assert "recall" in res.columns and "pr_auc" in res.columns
    assert res["recall"].notna().all()
