"""Entrenadores de los modelos del articulo: Arbol de Decision y Red Neuronal."""
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight


def train_decision_tree(X, y, max_depth=None, min_samples_leaf=1):
    """Arbol de decision con ganancia de informacion (entropia) y ponderacion de
    clases. Con ``max_depth=None`` crece sin poda y sobreajusta (hojas puras ->
    probabilidades 0/1, AUC-PR ~0.07); una profundidad acotada (p. ej. 8) actua
    como poda y mejora drasticamente la deteccion de brotes."""
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 class_weight="balanced", random_state=42)
    return clf.fit(X, y)


def _norm_params(X):
    """Estandarizacion z-score: media y desviacion estimadas SOLO en train
    (mejor que min-max para activaciones ReLU; evita fuga de informacion)."""
    mu = X.mean(axis=0).to_numpy(dtype=float)
    sd = X.std(axis=0).to_numpy(dtype=float)
    sd = np.where(sd == 0, 1.0, sd)
    return {"mean": mu, "std": sd}


def _apply_norm(X, norm):
    return (X.to_numpy(dtype=float) - norm["mean"]) / norm["std"]


def train_neural_net(X, y, epochs=60):
    """Red prealimentada 32-32 (ReLU) con salida sigmoide, entrenada con
    ponderacion de clases. Mejora clave frente a la version base: estandarizacion
    z-score en lugar de min-max. La estandarizacion (media 0, desviacion 1) es la
    normalizacion estandar para activaciones ReLU y, sin fuga (se estima solo en
    train), eleva el F1 de ~0.63 a ~0.70 respecto al escalado min-max."""
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
