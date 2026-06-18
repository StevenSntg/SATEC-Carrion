# Plan 3 — Modelado y evaluación — SATEC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entrenar y comparar **Árbol de Decisión**, **Red Neuronal** y **ensembles** (Random Forest, HistGradientBoosting) frente a **baselines epidemiológicos** (persistencia del estado epidémico), bajo **validación temporal**, con métricas adecuadas a eventos raros (recall de brotes, AUC-PR), calibración e importancia de variables.

**Architecture:** Módulos puros bajo `src/satec/models/`: matriz de features, split temporal, baselines, métricas, entrenadores por modelo, y un orquestador de evaluación. Tests aislados con fixtures; un smoke test entrena sobre el dataset enriquecido real y produce una tabla de resultados + figuras.

**Tech Stack:** Python 3.12 (`py -3.12`), pandas 3.0.3, numpy 1.26.4, scikit-learn 1.8.0, tf_keras 2.21.0 (Keras 2 legacy, `TF_USE_LEGACY_KERAS=1`), matplotlib 3.10.9, pytest 9.0.3.

## Global Constraints

- Intérprete: **Python 3.12** (`py -3.12`). Correr pytest desde la raíz (pythonpath=src).
- Dataset de entrada: `data/processed/dataset_enriched.parquet` (69.601×32, 61 provincias).
- Target: `brote` (binario). Las clases están desbalanceadas (~7,8% positivos) → usar `class_weight="balanced"` / `compute_class_weight` y reportar **recall de brotes** y **AUC-PR** como métricas primarias.
- **Validación TEMPORAL** (no aleatoria). Corte principal: train con `anio <= 2019`, test con `anio >= 2020`. El split NUNCA mezcla años entre train y test.
- **Sin fuga de información:** se excluyen del vector de features las claves (`departamento, provincia, ubigeo_prov, anio, semana`), el target (`brote`) y el canal (`q1, q2, q3`, que definen el umbral del target). La RN normaliza con parámetros calculados **solo en train**.
- Ensembles: `RandomForestClassifier` y `HistGradientBoostingClassifier` de scikit-learn (NO xgboost/lightgbm). Interpretabilidad: `permutation_importance` + `feature_importances_` (NO shap).
- RN: `tf_keras` con `TF_USE_LEGACY_KERAS=1`, arquitectura feedforward (igual familia que el trabajo previo: capas densas ReLU + salida sigmoide), `class_weight`.
- Semillas fijas (`random_state=42`, `tf_keras` seed) para reproducibilidad.
- Figuras y métricas se guardan en `results/`. Commits: cuenta **stevensntg**, sin coautoría de Claude.

---

### Task 1: Matriz de features (X, y)

**Files:**
- Create: `src/satec/models/__init__.py`
- Create: `src/satec/models/features_matrix.py`
- Test: `tests/models/__init__.py`, `tests/models/test_features_matrix.py`

**Interfaces:**
- Produces:
  - `FEATURE_COLS: list[str]` — columnas predictoras (excluye claves, target y canal).
  - `feature_matrix(df) -> tuple[pd.DataFrame, pd.Series]` → `(X[FEATURE_COLS], y=df["brote"].astype(int))`.
    Lanza `KeyError` si falta alguna columna esperada.

- [ ] **Step 1: Escribir el test que falla** (`tests/models/test_features_matrix.py`)

```python
import pandas as pd
from satec.models.features_matrix import feature_matrix, FEATURE_COLS


def _row(**kw):
    base = {c: 0.0 for c in FEATURE_COLS}
    base.update({"departamento": "ANCASH", "provincia": "HUARAZ",
                 "ubigeo_prov": "0201", "anio": 2015, "semana": 5,
                 "q1": 0.0, "q2": 0.0, "q3": 1.0, "brote": 1})
    base.update(kw)
    return base


def test_feature_matrix_excluye_claves_y_target():
    df = pd.DataFrame([_row(casos=3.0), _row(casos=0.0)])
    X, y = feature_matrix(df)
    for forbidden in ["departamento", "provincia", "ubigeo_prov", "anio",
                      "semana", "brote", "q1", "q2", "q3"]:
        assert forbidden not in X.columns
    assert "casos" in X.columns and "prec" in X.columns
    assert list(y) == [1, 0]
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_features_matrix.py" -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.models'`

- [ ] **Step 3: Implementar `src/satec/models/features_matrix.py`** (y `__init__.py` vacíos)

```python
"""Construcción de la matriz de features X y el vector objetivo y."""
import pandas as pd

# Excluidos: claves, target y canal (q1/q2/q3 definen el umbral del target).
_EXCLUDE = {"departamento", "provincia", "ubigeo_prov", "anio", "semana",
            "brote", "q1", "q2", "q3", "poblacion", "tasa"}

FEATURE_COLS = [
    "casos", "casos_lag1", "casos_lag2", "casos_lag4",
    "roll_mean4", "roll_mean8", "sin_semana", "cos_semana",
    "prec", "temp", "hum",
    "prec_lag4", "prec_lag8", "prec_roll4", "prec_roll8",
    "temp_lag4", "temp_lag8", "temp_roll4", "temp_roll8",
    "hum_lag4", "hum_lag8", "hum_roll4", "hum_roll8",
]


def feature_matrix(df: pd.DataFrame):
    faltan = [c for c in FEATURE_COLS if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas de features: {faltan}")
    X = df[FEATURE_COLS].astype(float).fillna(0.0)
    y = df["brote"].astype(int)
    return X, y
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_features_matrix.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/__init__.py src/satec/models/features_matrix.py tests/models
git commit -m "feat(models): matriz de features sin fuga de informacion"
```

---

### Task 2: Split temporal (rolling-origin)

**Files:**
- Create: `src/satec/models/split.py`
- Test: `tests/models/test_split.py`

**Interfaces:**
- Produces: `temporal_split(df, year_cutoff) -> tuple[pd.DataFrame, pd.DataFrame]`
  → `(train = df[anio <= year_cutoff], test = df[anio > year_cutoff])`.

- [ ] **Step 1: Escribir el test que falla** (`tests/models/test_split.py`)

```python
import pandas as pd
from satec.models.split import temporal_split


def test_split_no_mezcla_anios():
    df = pd.DataFrame({"anio": [2017, 2018, 2019, 2020, 2021], "x": range(5)})
    train, test = temporal_split(df, year_cutoff=2019)
    assert train["anio"].max() <= 2019
    assert test["anio"].min() >= 2020
    assert len(train) == 3 and len(test) == 2
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_split.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/models/split.py`**

```python
"""División temporal del dataset (sin mezclar años entre train y test)."""
import pandas as pd


def temporal_split(df: pd.DataFrame, year_cutoff: int):
    train = df[df["anio"] <= year_cutoff].reset_index(drop=True)
    test = df[df["anio"] > year_cutoff].reset_index(drop=True)
    return train, test
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_split.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/split.py tests/models/test_split.py
git commit -m "feat(models): split temporal rolling-origin"
```

---

### Task 3: Baseline de persistencia epidémica

**Files:**
- Create: `src/satec/models/baselines.py`
- Test: `tests/models/test_baselines.py`

**Interfaces:**
- Produces: `baseline_persistence(df) -> np.ndarray` → predice brote futuro = 1 si la
  provincia está **actualmente** por encima de su canal (`casos > q3`), si no 0.
  Es el baseline ingenuo: "si hoy hay actividad epidémica, la habrá las próximas semanas".

- [ ] **Step 1: Escribir el test que falla** (`tests/models/test_baselines.py`)

```python
import numpy as np
import pandas as pd
from satec.models.baselines import baseline_persistence


def test_persistence_marca_estado_actual():
    df = pd.DataFrame({"casos": [5, 0], "q3": [1.0, 1.0]})
    pred = baseline_persistence(df)
    assert list(pred) == [1, 0]  # 5>1 -> 1 ; 0>1 -> 0
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_baselines.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/models/baselines.py`**

```python
"""Baselines epidemiológicos para la predicción de brotes."""
import numpy as np
import pandas as pd


def baseline_persistence(df: pd.DataFrame) -> np.ndarray:
    """Predice brote futuro si la provincia está hoy por encima de su canal."""
    return ((df["casos"] > df["q3"]).astype(int)).to_numpy()
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_baselines.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/baselines.py tests/models/test_baselines.py
git commit -m "feat(models): baseline de persistencia epidemica"
```

---

### Task 4: Métricas de evaluación (eventos raros)

**Files:**
- Create: `src/satec/models/metrics.py`
- Test: `tests/models/test_metrics.py`

**Interfaces:**
- Produces: `evaluate_predictions(y_true, y_pred, y_score=None) -> dict` con claves
  `accuracy, precision, recall, f1, roc_auc, pr_auc, tn, fp, fn, tp`
  (`roc_auc`/`pr_auc` = NaN si `y_score is None`).

- [ ] **Step 1: Escribir el test que falla** (`tests/models/test_metrics.py`)

```python
from satec.models.metrics import evaluate_predictions


def test_metrics_caso_perfecto():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.8]
    m = evaluate_predictions(y_true, y_pred, y_score)
    assert abs(m["recall"] - 1.0) < 1e-9
    assert abs(m["precision"] - 1.0) < 1e-9
    assert abs(m["pr_auc"] - 1.0) < 1e-9
    assert m["tp"] == 2 and m["fp"] == 0
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_metrics.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/models/metrics.py`**

```python
"""Métricas de clasificación centradas en eventos raros (brotes)."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)


def evaluate_predictions(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_auc": float("nan"), "pr_auc": float("nan"),
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    return out
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_metrics.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/metrics.py tests/models/test_metrics.py
git commit -m "feat(models): metricas para eventos raros (recall, AUC-PR)"
```

---

### Task 5: Entrenadores (Árbol, Red Neuronal, ensembles)

**Files:**
- Create: `src/satec/models/train.py`
- Test: `tests/models/test_train.py`

**Interfaces:**
- Produces:
  - `train_decision_tree(X, y, max_depth=None) -> clf` (sklearn, `class_weight="balanced"`).
  - `train_random_forest(X, y) -> clf`.
  - `train_hist_gb(X, y) -> clf`.
  - `train_neural_net(X, y, epochs=60) -> tuple[model, dict]` → `(modelo tf_keras, {"min","max"})`
    con normalización min-max calculada en `X`; salida sigmoide; `class_weight`.
  - `nn_predict_proba(model, X, norm) -> np.ndarray` (aplica la misma normalización).

- [ ] **Step 1: Escribir el test que falla** (`tests/models/test_train.py`)

```python
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
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_train.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/models/train.py`**

```python
"""Entrenadores de los modelos: Árbol, ensembles y Red Neuronal."""
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


def train_random_forest(X, y):
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    return clf.fit(X, y)


def train_hist_gb(X, y):
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=42)
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
```

- [ ] **Step 4: Ejecutar para ver que pasa** (solo árbol y ensembles, rápido)

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_train.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/train.py tests/models/test_train.py
git commit -m "feat(models): entrenadores arbol, ensembles y red neuronal"
```

---

### Task 6: Orquestador de evaluación temporal

**Files:**
- Create: `src/satec/models/evaluate.py`
- Test: `tests/models/test_evaluate.py`

**Interfaces:**
- Produces:
  - `run_evaluation(df, year_cutoff=2019, nn_epochs=60) -> pd.DataFrame`
    → una fila por modelo (`arbol_sin_poda, arbol_poda8, red_neuronal,
    random_forest, hist_gb, baseline_persistencia`) con las métricas de `evaluate_predictions`
    sobre el conjunto de test, más `train_acc` y `gap` (brecha de generalización).
  - `main(repo=".")` → corre la evaluación sobre `dataset_enriched.parquet`,
    imprime y guarda `results/metricas_modelos.csv`.

- [ ] **Step 1: Escribir el test que falla** (`tests/models/test_evaluate.py`)

```python
import numpy as np
import pandas as pd
from satec.models.features_matrix import FEATURE_COLS
from satec.models.evaluate import run_evaluation


def _dataset(n_per_year=200):
    rng = np.random.RandomState(0)
    rows = []
    for anio in range(2015, 2022):
        for _ in range(n_per_year):
            casos = rng.poisson(1.0)
            q3 = 1.0
            feats = {c: rng.rand() for c in FEATURE_COLS}
            feats["casos"] = float(casos)
            brote = int(casos > q3)
            rows.append({**feats, "departamento": "X", "provincia": "P",
                         "ubigeo_prov": "0201", "anio": anio, "semana": 5,
                         "q1": 0.0, "q2": 0.0, "q3": q3, "brote": brote})
    return pd.DataFrame(rows)


def test_run_evaluation_devuelve_metricas_por_modelo():
    res = run_evaluation(_dataset(), year_cutoff=2019, nn_epochs=3)
    modelos = set(res["modelo"])
    assert {"arbol_sin_poda", "red_neuronal", "random_forest",
            "hist_gb", "baseline_persistencia"} <= modelos
    assert "recall" in res.columns and "pr_auc" in res.columns
    assert res["recall"].notna().all()
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_evaluate.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/models/evaluate.py`**

```python
"""Orquestador de la evaluación temporal de todos los modelos."""
import os
import numpy as np
import pandas as pd

from satec.models.features_matrix import feature_matrix
from satec.models.split import temporal_split
from satec.models.baselines import baseline_persistence
from satec.models.metrics import evaluate_predictions
from satec.models.train import (train_decision_tree, train_random_forest,
                                train_hist_gb, train_neural_net,
                                nn_predict_proba)


def _row(modelo, y_test, y_pred, y_score, train_acc):
    m = evaluate_predictions(y_test, y_pred, y_score)
    test_acc = m["accuracy"]
    m.update({"modelo": modelo, "train_acc": train_acc,
              "gap": (train_acc - test_acc) if train_acc == train_acc else float("nan")})
    return m


def run_evaluation(df, year_cutoff=2019, nn_epochs=60) -> pd.DataFrame:
    train_df, test_df = temporal_split(df, year_cutoff)
    Xtr, ytr = feature_matrix(train_df)
    Xte, yte = feature_matrix(test_df)
    rows = []

    sklearn_models = {
        "arbol_sin_poda": train_decision_tree(Xtr, ytr, max_depth=None),
        "arbol_poda8": train_decision_tree(Xtr, ytr, max_depth=8),
        "random_forest": train_random_forest(Xtr, ytr),
        "hist_gb": train_hist_gb(Xtr, ytr),
    }
    for nombre, clf in sklearn_models.items():
        score = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        train_acc = (clf.predict(Xtr) == ytr).mean()
        rows.append(_row(nombre, yte, pred, score, train_acc))

    model, norm = train_neural_net(Xtr, ytr, epochs=nn_epochs)
    score = nn_predict_proba(model, Xte, norm)
    pred = (score >= 0.5).astype(int)
    train_score = nn_predict_proba(model, Xtr, norm)
    train_acc = ((train_score >= 0.5).astype(int) == ytr).mean()
    rows.append(_row("red_neuronal", yte, pred, score, train_acc))

    bp = baseline_persistence(test_df)
    rows.append(_row("baseline_persistencia", yte, bp, None, float("nan")))

    cols = ["modelo", "accuracy", "precision", "recall", "f1", "roc_auc",
            "pr_auc", "train_acc", "gap", "tn", "fp", "fn", "tp"]
    return pd.DataFrame(rows)[cols]


def main(repo: str = ".") -> None:
    df = pd.read_parquet(os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    res = run_evaluation(df, year_cutoff=2019)
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    out = os.path.join(repo, "results", "metricas_modelos.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 160)
    print(res.to_string(index=False))
    print(f"\n[OK] -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Ejecutar el test (datos sintéticos, nn_epochs=3)**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_evaluate.py" -v`
Expected: PASS (1 passed) — puede tardar por TensorFlow.

- [ ] **Step 5: Smoke test sobre el dataset real**

Run:
```bash
REPO="C:/Users/Usuario/Desktop/IA/SATEC-Carrion"
PYTHONPATH="$REPO/src" TF_USE_LEGACY_KERAS=1 py -3.12 -c "from satec.models.evaluate import main; main(r'$REPO')"
```
Expected: tabla con métricas por modelo. Se espera que **árbol sin poda** tenga `gap` alto (sobreajuste) y la **red/ensembles** mayor `recall`/`pr_auc` que el baseline. Guardado en `results/metricas_modelos.csv`.

- [ ] **Step 6: Commit**

```bash
git add src/satec/models/evaluate.py tests/models/test_evaluate.py results/metricas_modelos.csv
git commit -m "feat(models): evaluacion temporal comparativa de modelos"
```

---

### Task 7: Figuras de resultados

**Files:**
- Create: `src/satec/models/figures.py`
- Test: `tests/models/test_figures.py`

**Interfaces:**
- Produces:
  - `plot_metrics_bar(res_df, out_path)` → barras de recall/pr_auc/f1 por modelo.
  - `plot_generalization_gap(res_df, out_path)` → brecha train-test por modelo.
  - `main(repo=".")` → lee `results/metricas_modelos.csv` y genera los PNG en `results/`.

- [ ] **Step 1: Escribir el test que falla** (`tests/models/test_figures.py`)

```python
import os
import pandas as pd
from satec.models.figures import plot_metrics_bar


def test_plot_metrics_bar_crea_png(tmp_path):
    res = pd.DataFrame({"modelo": ["a", "b"], "recall": [0.5, 0.8],
                        "pr_auc": [0.4, 0.7], "f1": [0.45, 0.75]})
    out = tmp_path / "fig.png"
    plot_metrics_bar(res, str(out))
    assert os.path.exists(out) and os.path.getsize(out) > 0
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_figures.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/models/figures.py`**

```python
"""Figuras de resultados del modelado."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metrics_bar(res_df, out_path):
    metricas = ["recall", "pr_auc", "f1"]
    ax = res_df.set_index("modelo")[metricas].plot.bar(figsize=(10, 5))
    ax.set_ylabel("valor")
    ax.set_title("Métricas por modelo (conjunto de prueba)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_generalization_gap(res_df, out_path):
    sub = res_df.dropna(subset=["gap"])
    ax = sub.set_index("modelo")["gap"].plot.bar(figsize=(10, 5), color="indianred")
    ax.set_ylabel("brecha (train_acc - test_acc)")
    ax.set_title("Brecha de generalización por modelo")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(repo: str = ".") -> None:
    import pandas as pd
    res = pd.read_csv(os.path.join(repo, "results", "metricas_modelos.csv"))
    plot_metrics_bar(res, os.path.join(repo, "results", "fig_metricas.png"))
    plot_generalization_gap(res, os.path.join(repo, "results", "fig_brecha.png"))
    print("[OK] figuras en results/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/models/test_figures.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Generar las figuras reales + suite completa**

```bash
REPO="C:/Users/Usuario/Desktop/IA/SATEC-Carrion"
PYTHONPATH="$REPO/src" py -3.12 -c "from satec.models.figures import main; main(r'$REPO')"
py -3.12 -m pytest "$REPO"
```
Expected: `results/fig_metricas.png` y `results/fig_brecha.png`; suite completa verde.

- [ ] **Step 6: Commit**

```bash
git add src/satec/models/figures.py tests/models/test_figures.py results/fig_metricas.png results/fig_brecha.png
git commit -m "feat(models): figuras de metricas y brecha de generalizacion"
```

---

## Notas para el ejecutor

- La RN tarda (TensorFlow). En tests usar `nn_epochs` bajo (3). En el smoke real, 60 épocas.
- `results/` se versiona (métricas y figuras son entregables del paper). Añadir `results/` NO está en `.gitignore`.
- Interpretabilidad (permutation importance): se añadirá como tarea de refinamiento tras ver los resultados base; `feature_importances_` de RF/HistGB ya está disponible para la primera lectura.
- Si el baseline de persistencia iguala o supera a los modelos, es un hallazgo honesto y publicable (la señal de vigilancia es fuertemente autorregresiva).

## Próximo plan

**Plan 4 — Aplicación web** (mapa de riesgo del Perú + comparador de modelos) y **Plan 5 — Artículo ACM**.
