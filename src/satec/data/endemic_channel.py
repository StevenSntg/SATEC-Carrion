"""Canal endemico causal: cuartiles por semana usando solo anios previos."""
import numpy as np
import pandas as pd


def compute_channel(panel: pd.DataFrame, n_ref: int = 5,
                    min_ref: int = 3) -> pd.DataFrame:
    panel = panel.sort_values(
        ["departamento", "provincia", "semana", "anio"]).reset_index(drop=True)
    q1 = np.full(len(panel), np.nan)
    q2 = np.full(len(panel), np.nan)
    q3 = np.full(len(panel), np.nan)

    grupos = panel.groupby(["departamento", "provincia", "semana"], sort=False)
    for _, idx in grupos.groups.items():
        idx = list(idx)
        sub = panel.loc[idx].sort_values("anio")
        casos = sub["casos"].to_numpy()
        posiciones = sub.index.to_numpy()
        for j in range(len(sub)):
            ref = casos[max(0, j - n_ref):j]  # anios estrictamente previos
            if len(ref) >= min_ref:
                pos = posiciones[j]
                q1[pos], q2[pos], q3[pos] = np.percentile(ref, [25, 50, 75])

    panel = panel.copy()
    panel["q1"], panel["q2"], panel["q3"] = q1, q2, q3
    return panel
