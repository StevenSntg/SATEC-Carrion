import numpy as np
import pandas as pd
from satec.models.train import (train_decision_tree, train_random_forest,
                                train_hist_gb)


def _xy(n=80):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.rand(n), "b": rng.rand(n)})
    y = (X["a"] + X["b"] > 1.0).astype(int)  # separable
    return X, y


def test_arbol_y_ensembles_aprenden_patron_separable():
    X, y = _xy()
    for train_fn in (train_decision_tree, train_random_forest, train_hist_gb):
        clf = train_fn(X, y)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.9
