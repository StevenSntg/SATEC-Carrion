"""Rejilla completa provincia x anio x semana con imputacion de ceros."""
import itertools
import pandas as pd

from satec.config import WEEKS


def build_full_grid(agg: pd.DataFrame) -> pd.DataFrame:
    provincias = agg[["departamento", "provincia"]].drop_duplicates()
    anios = sorted(agg["anio"].unique())
    combos = []
    for (_, dep, prov), anio, sem in itertools.product(
        provincias.itertuples(name=None), anios, WEEKS
    ):
        combos.append((dep, prov, anio, sem))
    return pd.DataFrame(
        combos, columns=["departamento", "provincia", "anio", "semana"])


def fill_zeros(agg: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    merged = grid.merge(
        agg, on=["departamento", "provincia", "anio", "semana"], how="left")
    merged["casos"] = merged["casos"].fillna(0).astype(int)
    return merged.sort_values(
        ["departamento", "provincia", "anio", "semana"]).reset_index(drop=True)
