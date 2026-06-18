"""Rezagos y medias moviles de variables climaticas, por provincia."""
import pandas as pd

_VARS = ["prec", "temp", "hum"]


def add_climate_features(panel: pd.DataFrame, lags=(4, 8),
                         windows=(4, 8)) -> pd.DataFrame:
    panel = panel.sort_values(
        ["ubigeo_prov", "anio", "semana"]).reset_index(drop=True)
    for var in _VARS:
        grp = panel.groupby("ubigeo_prov", sort=False)[var]
        for L in lags:
            panel[f"{var}_lag{L}"] = grp.shift(L).fillna(0)
        for W in windows:
            panel[f"{var}_roll{W}"] = grp.transform(
                lambda s: s.shift(1).rolling(W, min_periods=1).mean()).fillna(0)
    return panel
