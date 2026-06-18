"""Etiquetado del target de brote en la ventana [t+1, t+horizon]."""
import numpy as np
import pandas as pd


def label_outbreak(panel: pd.DataFrame, horizon: int = 4,
                   min_cases: int = 2) -> pd.DataFrame:
    panel = panel.sort_values(
        ["departamento", "provincia", "anio", "semana"]).reset_index(drop=True)
    brote = np.full(len(panel), np.nan)

    es_epi = ((panel["casos"] > panel["q3"]) &
              (panel["casos"] >= min_cases)).to_numpy()
    q3_na = panel["q3"].isna().to_numpy()

    grupos = panel.groupby(["departamento", "provincia"], sort=False)
    for _, idx in grupos.groups.items():
        idx = list(idx)
        for k in range(len(idx)):
            fin = k + horizon
            if fin >= len(idx):
                continue  # ventana futura incompleta -> NaN
            ventana = idx[k + 1:fin + 1]
            if any(q3_na[p] for p in ventana):
                continue  # sin canal en la ventana -> NaN
            brote[idx[k]] = 1.0 if any(es_epi[p] for p in ventana) else 0.0

    panel = panel.copy()
    panel["brote"] = pd.array(brote, dtype="Int64")
    return panel
