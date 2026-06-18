import pandas as pd
from satec.enrich.population_data import (load_province_population,
                                          attach_rate_constant)


def test_load_agrega_por_provincia(tmp_path):
    csv = tmp_path / "pob.csv"
    pd.DataFrame({"Ubigeo": ["010101", "010102"],
                  "Poblacion": [100, 50]}).to_csv(csv, index=False)
    out = load_province_population(str(csv))
    fila = out[out["ubigeo_prov"] == "0101"]
    assert int(fila["poblacion"].iloc[0]) == 150  # 100 + 50


def test_attach_rate_constant_por_100k():
    df = pd.DataFrame({"ubigeo_prov": ["0101"], "casos": [3]})
    pob = pd.DataFrame({"ubigeo_prov": ["0101"], "poblacion": [100000]})
    out = attach_rate_constant(df, pob)
    assert abs(out["tasa"].iloc[0] - 3.0) < 1e-9  # 3/100000*100000
