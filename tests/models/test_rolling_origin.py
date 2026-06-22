import numpy as np
import pandas as pd
from satec.models.features_matrix import FEATURE_COLS
from satec.models.rolling_origin import pooled_predictions, rolling_evaluation


def _ds(n_per_year=120):
    rng = np.random.RandomState(0)
    rows = []
    for anio in range(2012, 2020):
        for _ in range(n_per_year):
            feats = {c: rng.rand() for c in FEATURE_COLS}
            # señal: brote depende de 'roll_mean8'
            brote = int(feats["roll_mean8"] > 0.8)
            rows.append({**feats, "departamento": "X", "provincia": "P",
                         "ubigeo_prov": "0201", "anio": anio, "semana": 5,
                         "q1": 0.0, "q2": 0.0, "q3": 1.0, "casos": 0.0,
                         "brote": brote})
    return pd.DataFrame(rows)


def test_pooled_predictions_respeta_anios_de_prueba():
    from sklearn.tree import DecisionTreeClassifier
    df = _ds()
    build = lambda X, y: DecisionTreeClassifier(random_state=0).fit(X, y)
    score = lambda m, X: m.predict_proba(X)[:, 1]
    out = pooled_predictions(df, build, score, test_years=range(2015, 2020),
                             val_min_pos=1)
    # 5 años de prueba × 120 filas
    assert len(out["y_true"]) == 5 * 120
    assert set(np.unique(out["y_pred"])) <= {0, 1}


def test_rolling_evaluation_incluye_seis_modelos():
    df = _ds()
    res = rolling_evaluation(df, test_years=range(2015, 2020), nn_epochs=2)
    assert {"red_neuronal", "random_forest", "gradient_boosting",
            "arbol_poda8", "arbol_sin_poda", "baseline_persistencia"} <= set(res["modelo"])
    for col in ["f1", "f2", "especificidad", "pr_auc", "umbral"]:
        assert col in res.columns
