"""Poblacion provincial (censo INEI 2017) y tasas por 100.000 habitantes."""
import pandas as pd


def load_province_population(csv_path: str) -> pd.DataFrame:
    """Agrega la poblacion distrital del censo 2017 a nivel provincia.

    El CSV de origen tiene columnas Ubigeo (distrital, 6 digitos) y Poblacion
    (con separador de miles). Devuelve ["ubigeo_prov", "poblacion"].
    """
    df = pd.read_csv(csv_path, thousands=",")
    df["ubigeo_prov"] = df["Ubigeo"].astype(str).str.zfill(6).str[:4]
    out = (df.groupby("ubigeo_prov")["Poblacion"].sum()
             .reset_index().rename(columns={"Poblacion": "poblacion"}))
    out["poblacion"] = out["poblacion"].astype(int)
    return out


def attach_rate_constant(df: pd.DataFrame, pob_df: pd.DataFrame) -> pd.DataFrame:
    """Une poblacion (constante por provincia) y calcula la tasa por 100k.

    Poblacion de referencia (censo 2017); no captura la dinamica poblacional
    interanual -> declarado como limitacion en el paper.
    """
    out = df.merge(pob_df, on="ubigeo_prov", how="left")
    out["tasa"] = out["casos"] / out["poblacion"] * 100000
    return out
