import numpy as np
import pandas as pd
from satec.models.train import train_decision_tree


def _xy(n=80):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.rand(n), "b": rng.rand(n)})
    y = (X["a"] + X["b"] > 1.0).astype(int)  # separable
    return X, y


def test_arbol_aprende_patron_separable():
    X, y = _xy()
    clf = train_decision_tree(X, y)
    acc = (clf.predict(X) == y).mean()
    assert acc > 0.9
