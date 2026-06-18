import pandas as pd
from satec.enrich.build_enriched import attach_ubigeo


def test_attach_ubigeo_por_nombre():
    dataset = pd.DataFrame({"departamento": ["ANCASH"], "provincia": ["HUARAZ"],
                            "anio": [2015], "semana": [5], "casos": [3]})
    umap = pd.DataFrame({"departamento": ["ANCASH"], "provincia": ["HUARAZ"],
                         "ubigeo_prov": ["0201"]})
    out = attach_ubigeo(dataset, umap)
    assert out["ubigeo_prov"].iloc[0] == "0201"
