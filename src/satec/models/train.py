"""Entrenadores de los modelos: Arbol, ensembles y Red Neuronal."""
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.utils.class_weight import compute_class_weight


def train_decision_tree(X, y, max_depth=None):
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth,
                                 class_weight="balanced", random_state=42)
    return clf.fit(X, y)


def train_random_forest(X, y, n_estimators=300, max_depth=None):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", random_state=42, n_jobs=-1)
    return clf.fit(X, y)


def train_gradient_boosting(X, y, max_iter=500):
    """Gradient Boosting por histogramas, regularizado (afinado para el panel
    de Carrión: árboles someros + L2 + hojas grandes reducen el sobreajuste)."""
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.05, max_depth=3,
        l2_regularization=0.1, min_samples_leaf=50,
        class_weight="balanced", random_state=42)
    return clf.fit(X, y)


def _norm_params(X):
    xmin = X.min(axis=0).to_numpy(dtype=float)
    xmax = X.max(axis=0).to_numpy(dtype=float)
    rng = np.where((xmax - xmin) == 0, 1.0, xmax - xmin)
    return {"min": xmin, "rng": rng}


def _apply_norm(X, norm):
    return (X.to_numpy(dtype=float) - norm["min"]) / norm["rng"]


def train_neural_net(X, y, epochs=60):
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tf_keras as keras
    from tf_keras import layers
    keras.utils.set_random_seed(42)

    norm = _norm_params(X)
    Xn = _apply_norm(X, norm)
    classes = np.array([0, 1])
    w = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {0: float(w[0]), 1: float(w[1])}

    model = keras.Sequential([
        layers.Dense(32, input_dim=Xn.shape[1], activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.fit(Xn, np.asarray(y), epochs=epochs, batch_size=256, verbose=0,
              class_weight=class_weight)
    return model, norm


def nn_predict_proba(model, X, norm) -> np.ndarray:
    Xn = _apply_norm(X, norm)
    return model.predict(Xn, verbose=0).ravel()
