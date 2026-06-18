import pandas as pd
from satec.data.target import label_outbreak


def _panel_con_canal():
    # provincia unica, semanas 1..10 de un anio, con q3 constante = 1
    rows = []
    casos = [0, 0, 0, 0, 5, 0, 0, 0, 0, 0]  # pico en semana 5
    for i, c in enumerate(casos, start=1):
        rows.append(("ANCASH", "HUARAZ", 2015, i, c, 0.0, 0.0, 1.0))
    return pd.DataFrame(rows, columns=[
        "departamento", "provincia", "anio", "semana", "casos",
        "q1", "q2", "q3"])


def test_target_marca_brote_en_ventana_previa():
    out = label_outbreak(_panel_con_canal(), horizon=4, min_cases=2)
    # semana 1: ventana [2..5] incluye el pico (5 > q3=1 y >=2) -> brote=1
    assert int(out[out["semana"] == 1]["brote"].iloc[0]) == 1
    # semana 5: ventana [6..9] sin picos -> brote=0
    assert int(out[out["semana"] == 5]["brote"].iloc[0]) == 0


def test_target_nulo_si_ventana_incompleta():
    out = label_outbreak(_panel_con_canal(), horizon=4, min_cases=2)
    # semanas 7..10 no tienen 4 semanas futuras dentro del panel
    assert out[out["semana"] == 10]["brote"].isna().all()
