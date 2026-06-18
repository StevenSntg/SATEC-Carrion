"""Agregacion de casos individuales a conteos provincia x anio x semana."""
import pandas as pd

KEYS = ["departamento", "provincia", "anio", "semana"]


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(KEYS, as_index=False)
             .size()
             .rename(columns={"size": "casos"}))
    agg["casos"] = agg["casos"].astype(int)
    return agg[KEYS + ["casos"]]
