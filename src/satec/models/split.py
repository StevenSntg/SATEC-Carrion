"""Division temporal del dataset (sin mezclar anios entre train y test)."""
import pandas as pd


def temporal_split(df: pd.DataFrame, year_cutoff: int):
    train = df[df["anio"] <= year_cutoff].reset_index(drop=True)
    test = df[df["anio"] > year_cutoff].reset_index(drop=True)
    return train, test
