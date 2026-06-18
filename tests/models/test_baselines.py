import numpy as np
import pandas as pd
from satec.models.baselines import baseline_persistence


def test_persistence_marca_estado_actual():
    df = pd.DataFrame({"casos": [5, 0], "q3": [1.0, 1.0]})
    pred = baseline_persistence(df)
    assert list(pred) == [1, 0]  # 5>1 -> 1 ; 0>1 -> 0
