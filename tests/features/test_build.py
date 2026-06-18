import numpy as np
import pandas as pd
from satec.features.build import add_features


def _panel():
    rows = []
    for i, sem in enumerate(range(1, 11)):
        rows.append(("ANCASH", "HUARAZ", 2015, sem, i))  # casos = 0..9
    return pd.DataFrame(
        rows, columns=["departamento", "provincia", "anio", "semana", "casos"])


def test_lag_features():
    out = add_features(_panel(), lags=(1, 2, 4), windows=(4, 8))
    fila = out[out["semana"] == 5].iloc[0]
    assert fila["casos"] == 4
    assert fila["casos_lag1"] == 3
    assert fila["casos_lag4"] == 0  # casos de la semana 1


def test_seasonal_features_en_rango():
    out = add_features(_panel())
    assert out["sin_semana"].between(-1, 1).all()
    assert out["cos_semana"].between(-1, 1).all()
