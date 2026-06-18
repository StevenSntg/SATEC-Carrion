import pandas as pd
from satec.web_export.risk_map import latest_risk_table


def test_latest_toma_ultima_semana_por_provincia():
    df = pd.DataFrame({
        "ubigeo_prov": ["0201", "0201", "0301"],
        "anio": [2024, 2024, 2024], "semana": [10, 48, 48],
        "casos": [1, 5, 2], "q3": [1.0, 1.0, 1.0]})
    out = latest_risk_table(df)
    fila = out[out["ubigeo_prov"] == "0201"]
    assert fila["semana"].iloc[0] == 48 and fila["casos"].iloc[0] == 5
