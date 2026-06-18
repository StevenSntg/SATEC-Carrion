from satec.data.load import clean
from satec.data.aggregate import aggregate_weekly


def test_aggregate_cuenta_casos(sample_raw):
    agg = aggregate_weekly(clean(sample_raw))
    fila = agg[(agg["provincia"] == "HUARAZ") & (agg["anio"] == 2010)
               & (agg["semana"] == 5)]
    assert int(fila["casos"].iloc[0]) == 2  # dos casos en HUARAZ 2010 s5


def test_aggregate_columnas(sample_raw):
    agg = aggregate_weekly(clean(sample_raw))
    assert list(agg.columns) == [
        "departamento", "provincia", "anio", "semana", "casos"]
