from satec.config import WEEKS
from satec.data.load import clean
from satec.data.aggregate import aggregate_weekly
from satec.data.grid import build_full_grid, fill_zeros


def test_grid_es_producto_cartesiano(sample_raw):
    agg = aggregate_weekly(clean(sample_raw))
    grid = build_full_grid(agg)
    n_prov = agg[["departamento", "provincia"]].drop_duplicates().shape[0]
    n_anios = agg["anio"].nunique()
    assert len(grid) == n_prov * n_anios * len(WEEKS)


def test_fill_zeros_imputa_y_conserva_conteos(sample_raw):
    agg = aggregate_weekly(clean(sample_raw))
    full = fill_zeros(agg, build_full_grid(agg))
    # HUARAZ 2010 s5 conserva 2 casos
    f = full[(full["provincia"] == "HUARAZ") & (full["anio"] == 2010)
             & (full["semana"] == 5)]
    assert int(f["casos"].iloc[0]) == 2
    # una semana sin registros queda en 0, sin nulos
    assert full["casos"].isna().sum() == 0
    assert (full["casos"] == 0).any()
