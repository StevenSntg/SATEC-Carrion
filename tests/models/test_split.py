import pandas as pd
from satec.models.split import temporal_split


def test_split_no_mezcla_anios():
    df = pd.DataFrame({"anio": [2017, 2018, 2019, 2020, 2021], "x": range(5)})
    train, test = temporal_split(df, year_cutoff=2019)
    assert train["anio"].max() <= 2019
    assert test["anio"].min() >= 2020
    assert len(train) == 3 and len(test) == 2
