"""Baselines epidemiologicos para la prediccion de brotes."""
import numpy as np
import pandas as pd


def baseline_persistence(df: pd.DataFrame) -> np.ndarray:
    """Predice brote futuro si la provincia esta hoy por encima de su canal."""
    return ((df["casos"] > df["q3"]).astype(int)).to_numpy()
