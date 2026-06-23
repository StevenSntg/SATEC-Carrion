import numpy as np
import pandas as pd
from satec.models.train import train_decision_tree, train_random_forest, train_gradient_boosting


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


def test_random_forest_aprende_patron_separable():
    X, y = _xy()
    clf = train_random_forest(X, y, n_estimators=50)
    assert (clf.predict(X) == y).mean() > 0.9
    assert clf.predict_proba(X).shape[1] == 2


def test_gradient_boosting_aprende_patron_separable():
    # El GB afinado usa min_samples_leaf=50, por lo que requiere más muestras
    # que el patrón sintético mínimo para poder dividir.
    X, y = _xy(600)
    clf = train_gradient_boosting(X, y, max_iter=300)
    assert (clf.predict(X) == y).mean() > 0.85
