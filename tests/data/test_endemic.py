import pandas as pd
from satec.data.endemic import select_endemic_provinces


def _agg(rows):
    return pd.DataFrame(
        rows, columns=["departamento", "provincia", "anio", "semana", "casos"])


def test_endemic_filtra_por_umbrales():
    agg = _agg([
        ("ANCASH", "HUARAZ", 2010, 5, 8),
        ("ANCASH", "HUARAZ", 2011, 5, 8),
        ("ANCASH", "HUARAZ", 2012, 5, 8),   # 24 casos en 3 anios -> endemica
        ("LIMA", "HUARAL", 2010, 8, 1),     # 1 caso, 1 anio -> NO endemica
    ])
    out = select_endemic_provinces(agg, min_casos=10, min_anios=3)
    provincias = set(out["provincia"])
    assert "HUARAZ" in provincias
    assert "HUARAL" not in provincias
