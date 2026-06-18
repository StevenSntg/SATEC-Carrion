import numpy as np
import pandas as pd
from satec.data.endemic_channel import compute_channel


def _panel():
    # Una provincia, semana 5 fija, casos crecientes por anio
    rows = []
    for i, anio in enumerate(range(2010, 2017)):
        rows.append(("ANCASH", "HUARAZ", anio, 5, i))  # casos = 0..6
    return pd.DataFrame(
        rows, columns=["departamento", "provincia", "anio", "semana", "casos"])


def test_channel_nan_si_falta_historia():
    out = compute_channel(_panel(), n_ref=5, min_ref=3)
    # 2010 y 2011 no tienen >=3 anios previos
    early = out[out["anio"].isin([2010, 2011])]
    assert early[["q1", "q2", "q3"]].isna().all().all()


def test_channel_usa_solo_anios_previos():
    out = compute_channel(_panel(), n_ref=5, min_ref=3)
    # Para 2015 (casos previos en s5: 2010..2014 -> 0,1,2,3,4), Q2 = mediana = 2
    fila = out[(out["anio"] == 2015) & (out["semana"] == 5)]
    assert abs(float(fila["q2"].iloc[0]) - 2.0) < 1e-9
