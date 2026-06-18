import pandas as pd
from satec.data.ubigeo import province_ubigeo_map


def test_ubigeo_prov_modal():
    raw = pd.DataFrame({
        "departamento": ["ANCASH", "ANCASH", "ANCASH"],
        "provincia": ["HUARAZ", "HUARAZ", "HUARAZ"],
        "ubigeo": ["020101", "020102", "020101"],  # prov 0201 modal
    })
    out = province_ubigeo_map(raw)
    fila = out[(out["provincia"] == "HUARAZ")]
    assert fila["ubigeo_prov"].iloc[0] == "0201"
