"""Homologacion de (departamento, provincia) -> ubigeo de provincia."""
import pandas as pd


def province_ubigeo_map(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["departamento"] = df["departamento"].astype(str).str.strip().str.upper()
    df["provincia"] = df["provincia"].astype(str).str.strip().str.upper()
    df["ubigeo_prov"] = df["ubigeo"].astype(str).str.zfill(6).str[:4]

    def _modal(s):
        return s.mode().iloc[0]

    out = (df.groupby(["departamento", "provincia"])["ubigeo_prov"]
             .agg(_modal).reset_index())
    return out
