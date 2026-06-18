"""Carga y limpieza del CSV crudo de vigilancia del MINSA."""
import pandas as pd

_FASE_MAP = {
    "ENFERMEDAD DE CARRION AGUDA": "AGUDA",
    "ENFERMEDAD DE CARRION ERUPTIVA": "ERUPTIVA",
}


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _edad_a_anios(edad, tipo_edad) -> float:
    t = str(tipo_edad).strip().upper()
    edad = float(edad)
    if t == "M":
        return edad / 12.0
    if t == "D":
        return edad / 365.0
    return edad


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fase"] = df["enfermedad"].map(_FASE_MAP)
    df = df.dropna(subset=["fase"])  # descarta filas fuera de las dos fases

    df["anio"] = df["ano"].astype(int)
    df["semana"] = df["semana"].astype(int).clip(upper=52)  # 53 -> 52

    df["edad_anios"] = df.apply(
        lambda r: _edad_a_anios(r["edad"], r["tipo_edad"]), axis=1
    )
    df["sexo"] = df["sexo"].astype(str).str.strip().str.upper()
    df["departamento"] = df["departamento"].astype(str).str.strip().str.upper()
    df["provincia"] = df["provincia"].astype(str).str.strip().str.upper()
    df["ubigeo"] = df["ubigeo"].astype(str).str.strip()

    cols = ["departamento", "provincia", "ubigeo", "anio", "semana",
            "fase", "edad_anios", "sexo"]
    return df[cols].reset_index(drop=True)
