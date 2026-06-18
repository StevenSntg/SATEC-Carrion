"""Caracteristicas temporales por provincia."""
import numpy as np
import pandas as pd


def add_features(panel: pd.DataFrame, lags=(1, 2, 4),
                 windows=(4, 8)) -> pd.DataFrame:
    panel = panel.sort_values(
        ["departamento", "provincia", "anio", "semana"]).reset_index(drop=True)
    grp = panel.groupby(["departamento", "provincia"], sort=False)["casos"]

    for L in lags:
        panel[f"casos_lag{L}"] = grp.shift(L).fillna(0)
    for W in windows:
        # rolling por grupo (transform realinea al panel; evita cruzar provincias)
        panel[f"roll_mean{W}"] = grp.transform(
            lambda s: s.shift(1).rolling(W, min_periods=1).mean()).fillna(0)

    ang = 2 * np.pi * panel["semana"] / 52.0
    panel["sin_semana"] = np.sin(ang)
    panel["cos_semana"] = np.cos(ang)
    return panel
