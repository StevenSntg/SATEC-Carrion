import pandas as pd
from satec.models.features_matrix import feature_matrix, FEATURE_COLS


def _row(**kw):
    base = {c: 0.0 for c in FEATURE_COLS}
    base.update({"departamento": "ANCASH", "provincia": "HUARAZ",
                 "ubigeo_prov": "0201", "anio": 2015, "semana": 5,
                 "q1": 0.0, "q2": 0.0, "q3": 1.0, "brote": 1})
    base.update(kw)
    return base


def test_feature_matrix_excluye_claves_y_target():
    df = pd.DataFrame([_row(casos=3.0), _row(casos=0.0, brote=0)])
    X, y = feature_matrix(df)
    for forbidden in ["departamento", "provincia", "ubigeo_prov", "anio",
                      "semana", "brote", "q1", "q2", "q3"]:
        assert forbidden not in X.columns
    assert "casos" in X.columns and "prec" in X.columns
    assert list(y) == [1, 0]
