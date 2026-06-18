import pandas as pd
from satec.enrich.climate_features import add_climate_features


def _panel():
    rows = []
    for i, sem in enumerate(range(1, 13)):
        rows.append(("0201", 2015, sem, float(i), 10.0 + i, 60.0))
    return pd.DataFrame(rows, columns=[
        "ubigeo_prov", "anio", "semana", "prec", "temp", "hum"])


def test_lag4_de_precipitacion():
    out = add_climate_features(_panel(), lags=(4, 8), windows=(4, 8))
    fila = out[out["semana"] == 9].iloc[0]
    # prec en semana 9 (i=8) = 8.0; prec_lag4 = prec semana 5 (i=4) = 4.0
    assert abs(fila["prec_lag4"] - 4.0) < 1e-9
