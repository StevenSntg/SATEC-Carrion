"""Seleccion data-driven de provincias endemicas."""
import pandas as pd


def select_endemic_provinces(
    agg: pd.DataFrame, min_casos: int = 10, min_anios: int = 3
) -> pd.DataFrame:
    con_casos = agg[agg["casos"] > 0]
    resumen = con_casos.groupby(["departamento", "provincia"]).agg(
        total_casos=("casos", "sum"),
        anios_con_casos=("anio", "nunique"),
    ).reset_index()
    sel = resumen[(resumen["total_casos"] >= min_casos)
                  & (resumen["anios_con_casos"] >= min_anios)]
    return sel[["departamento", "provincia"]].reset_index(drop=True)
