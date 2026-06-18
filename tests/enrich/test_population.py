import pandas as pd
from satec.enrich.population import attach_rate


def test_attach_rate_calcula_por_100k():
    panel = pd.DataFrame({"ubigeo_prov": ["0201"], "anio": [2015], "casos": [5]})
    pob = pd.DataFrame({"ubigeo_prov": ["0201"], "anio": [2015],
                        "poblacion": [100000]})
    out = attach_rate(panel, pob)
    assert abs(out["tasa"].iloc[0] - 5.0) < 1e-9  # 5/100000*100000 = 5
