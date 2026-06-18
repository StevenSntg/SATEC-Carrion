import pandas as pd
from satec.data.load import clean


def test_clean_normaliza_semana_53_a_52(sample_raw):
    out = clean(sample_raw)
    assert out["semana"].max() <= 52
    assert (out["semana"] == 52).sum() == 1  # la fila de semana 53 pasa a 52


def test_clean_columnas_y_tipos(sample_raw):
    out = clean(sample_raw)
    assert list(out.columns) == [
        "departamento", "provincia", "ubigeo", "anio", "semana",
        "fase", "edad_anios", "sexo",
    ]
    assert out["anio"].dtype.kind == "i"
    assert set(out["fase"].unique()) <= {"AGUDA", "ERUPTIVA"}


def test_clean_convierte_edad_en_meses_a_anios(sample_raw):
    out = clean(sample_raw)
    fila_meses = out[(out["provincia"] == "HUARAL")]
    assert abs(float(fila_meses["edad_anios"].iloc[0]) - 1.0) < 1e-9
