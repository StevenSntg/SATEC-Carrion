"""Construccion de la matriz de features X y el vector objetivo y."""
import pandas as pd

# Excluidos: claves, target y canal (q1/q2/q3 definen el umbral del target).
_EXCLUDE = {"departamento", "provincia", "ubigeo_prov", "anio", "semana",
            "brote", "q1", "q2", "q3", "poblacion", "tasa"}

FEATURE_COLS = [
    "casos", "casos_lag1", "casos_lag2", "casos_lag4",
    "roll_mean4", "roll_mean8", "sin_semana", "cos_semana",
    "prec", "temp", "hum",
    "prec_lag4", "prec_lag8", "prec_roll4", "prec_roll8",
    "temp_lag4", "temp_lag8", "temp_roll4", "temp_roll8",
    "hum_lag4", "hum_lag8", "hum_roll4", "hum_roll8",
    "tasa",
]


def feature_matrix(df: pd.DataFrame):
    faltan = [c for c in FEATURE_COLS if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas de features: {faltan}")
    X = df[FEATURE_COLS].astype(float).fillna(0.0)
    y = df["brote"].astype(int)
    return X, y
