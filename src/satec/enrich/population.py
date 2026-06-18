"""Union de poblacion provincial y calculo de tasas por 100.000 hab."""
import pandas as pd


def attach_rate(panel: pd.DataFrame, pob_df: pd.DataFrame) -> pd.DataFrame:
    out = panel.merge(pob_df, on=["ubigeo_prov", "anio"], how="left")
    out["tasa"] = out["casos"] / out["poblacion"] * 100000
    return out
