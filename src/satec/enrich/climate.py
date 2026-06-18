"""Parseo y agregacion semanal de datos climaticos de NASA POWER."""
import numpy as np
import pandas as pd

_PARAMS = {"PRECTOTCORR": "prec", "T2M": "temp", "RH2M": "hum"}
_FILL = -999.0


def parse_power_json(payload: dict) -> pd.DataFrame:
    param = payload["properties"]["parameter"]
    fechas = sorted(param["PRECTOTCORR"].keys())
    data = {"fecha": pd.to_datetime(fechas, format="%Y%m%d")}
    for src, dst in _PARAMS.items():
        serie = [param[src][f] for f in fechas]
        data[dst] = [np.nan if v == _FILL else v for v in serie]
    return pd.DataFrame(data)


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    iso = df["fecha"].dt.isocalendar()
    df["anio"] = iso["year"].astype(int)
    df["semana"] = iso["week"].astype(int).clip(upper=52)
    agg = df.groupby(["anio", "semana"], as_index=False).agg(
        prec=("prec", "sum"), temp=("temp", "mean"), hum=("hum", "mean"))
    return agg
