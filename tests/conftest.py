import pandas as pd
import pytest


@pytest.fixture
def sample_raw():
    """DataFrame mínimo que imita el CSV crudo del MINSA."""
    rows = [
        # departamento, provincia, distrito, localidad, enfermedad, ano, semana,
        # diagnostic, diresa, ubigeo, localcod, edad, tipo_edad, sexo
        ("ANCASH", "HUARAZ", "HUARAZ", "X", "ENFERMEDAD DE CARRION AGUDA", 2010, 5,
         "A44.0", "02", "020101", "", 19, "A", "M"),
        ("ANCASH", "HUARAZ", "HUARAZ", "X", "ENFERMEDAD DE CARRION ERUPTIVA", 2010, 5,
         "A44.1", "02", "020101", "", 4, "A", "F"),
        ("ANCASH", "HUARAZ", "HUARAZ", "X", "ENFERMEDAD DE CARRION AGUDA", 2010, 53,
         "A44.0", "02", "020101", "", 30, "A", "M"),
        ("LIMA", "HUARAL", "IHUARI", "Y", "ENFERMEDAD DE CARRION AGUDA", 2011, 8,
         "A44.0", "42", "150606", "", 12, "M", "F"),  # 12 meses -> 1.0 año
    ]
    return pd.DataFrame(rows, columns=[
        "departamento", "provincia", "distrito", "localidad", "enfermedad",
        "ano", "semana", "diagnostic", "diresa", "ubigeo", "localcod",
        "edad", "tipo_edad", "sexo",
    ])
