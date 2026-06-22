# Corrección de modelos y artículo SATEC — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevar el F1 de forma metodológicamente legítima (validación de origen móvil + umbral óptimo), añadir Random Forest y Gradient Boosting, y resolver las 19 observaciones del profesor, dejando el artículo ACM regenerado con rigor de revista científica.

**Architecture:** Se extiende el paquete `satec.models` con métricas (F2, especificidad), dos entrenadores nuevos (RF, GB), un selector de umbral y un módulo de validación de origen móvil. El orquestador de evaluación produce un `metricas_modelos.csv` ampliado del que se regeneran todas las figuras. El artículo (`paper/SATEC_articulo_ACM.md`) se reescribe en las secciones afectadas, se renumeran las referencias y se regenera el `.docx` con `build_acm.py`.

**Tech Stack:** Python 3.12, scikit-learn, tf-keras/TensorFlow, matplotlib, pandas, numpy, python-docx + latex2mathml + XSLT de Office, nbformat, pytest.

## Global Constraints

- Python 3.12; semilla global `random_state=42` / `set_random_seed(42)` en todo modelo.
- Validación **temporal estricta sin fuga**: el año de prueba nunca interviene en el entrenamiento ni en la elección del umbral.
- Canal endémico (`q1/q2/q3`) y target (`brote`) ya son causales en `data/processed/dataset_enriched.parquet`; **no recalcular**.
- Las 24 features son las de `FEATURE_COLS` en `src/satec/models/features_matrix.py`; no alterar el conjunto.
- Figuras: estilo de `src/satec/models/paper_style.py` (paleta Okabe-Ito, serif, 300 dpi, sin título embebido).
- Commits con la cuenta **stevensntg**; **NUNCA** añadir `Co-Authored-By: Claude`. Estilo de mensaje: conventional commits en español (`feat:`, `fix:`, `docs:`, `test:`).
- Toda cifra que entre al paper debe provenir de un artefacto reproducible (`results/*.csv`) o validarse con el usuario (comparación con literatura).
- `pytest -q` debe quedar en verde tras cada tarea de código.

## File Structure

**Crear:**
- `src/satec/models/thresholds.py` — selección de umbral óptimo por F-beta.
- `src/satec/models/rolling_origin.py` — validación de origen móvil (entrena por corte, elige umbral en validación, agrupa predicciones).
- `src/satec/models/figure_architecture.py` — figura de la arquitectura de la RN.
- `paper/cite_lint.py` — validador de linealidad de citas IEEE/ACM.
- `tests/models/test_thresholds.py`, `tests/models/test_rolling_origin.py`, `tests/models/test_figure_architecture.py`, `tests/paper/test_cite_lint.py`, `tests/paper/__init__.py`.

**Modificar:**
- `requirements.txt` — añadir dependencias usadas.
- `src/satec/models/metrics.py` — F2 y especificidad.
- `src/satec/models/train.py` — `train_random_forest`, `train_gradient_boosting`.
- `src/satec/models/evaluate.py` — orquestación con 6 modelos + origen móvil + umbral.
- `src/satec/models/figures.py`, `figures_extra.py` — etiquetas numéricas, RF en matrices, 6 modelos.
- `src/satec/models/paper_style.py` — nombres/colores de RF y GB.
- `src/satec/models/run_interpret.py` — calibración sobre el pool de origen móvil.
- `src/satec/web_export/export_models.py` — re-exportar (y RF opcional).
- `notebooks/build_notebooks.py` — notebook 03 (RF/GB) y actualización de 01/02.
- `paper/SATEC_articulo_ACM.md` — texto, Tabla 1, referencias, keywords, agradecimientos.
- `paper/build_acm.py` — sin sangría, pie de fuente en figuras.

---

## FASE 0 — Reproducibilidad

### Task 1: Declarar dependencias reales

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Produces: entorno instalable con sklearn, tf-keras, tensorflow, tensorflowjs.

- [ ] **Step 1: Escribir `requirements.txt`**

```
pandas==3.0.3
numpy==1.26.4
pytest==9.0.3
pyarrow
scikit-learn>=1.5
tf-keras
tensorflow
tensorflowjs
matplotlib
nbformat
python-docx
latex2mathml
lxml
pillow
```

- [ ] **Step 2: Verificar import de ensembles y versión de sklearn**

Run: `python -c "import sklearn; from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier; print(sklearn.__version__)"`
Expected: imprime una versión ≥ 1.5 sin error.

> Si la versión es < 1.5, `HistGradientBoostingClassifier` no acepta `class_weight`; en ese caso la Task 3 usará `sample_weight` (se indica allí).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build: declara scikit-learn, tf-keras y deps del paper en requirements"
```

---

## FASE 1 — Modelado y evaluación (resuelve el F1)

### Task 2: Métricas F2 y especificidad

**Files:**
- Modify: `src/satec/models/metrics.py`
- Test: `tests/models/test_metrics.py`

**Interfaces:**
- Consumes: `evaluate_predictions(y_true, y_pred, y_score=None) -> dict`.
- Produces: el dict incluye además `"f2"` y `"especificidad"`.

- [ ] **Step 1: Añadir el test que falla** (append a `tests/models/test_metrics.py`)

```python
def test_metrics_incluye_f2_y_especificidad():
    # 2 positivos: 1 detectado (tp) y 1 perdido (fn); 1 falsa alarma (fp)
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    m = evaluate_predictions(y_true, y_pred)
    assert "f2" in m and "especificidad" in m
    # especificidad = tn/(tn+fp) = 1/2
    assert abs(m["especificidad"] - 0.5) < 1e-9
    # F2 pondera más el recall que F1; con recall=0.5 y precision=0.5, F2=0.5
    assert abs(m["f2"] - 0.5) < 1e-9
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/models/test_metrics.py::test_metrics_incluye_f2_y_especificidad -v`
Expected: FAIL con `KeyError: 'f2'`.

- [ ] **Step 3: Implementar**

En `src/satec/models/metrics.py`, importar `fbeta_score` y añadir las claves dentro de `evaluate_predictions`, después de calcular `tn, fp, fn, tp`:

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             brier_score_loss)
```

Y dentro del dict `out` (junto a `"f1"`):

```python
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "especificidad": (tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/models/test_metrics.py -v`
Expected: PASS (los dos tests).

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/metrics.py tests/models/test_metrics.py
git commit -m "feat(models): añade F2 y especificidad a las métricas"
```

### Task 3: Entrenadores Random Forest y Gradient Boosting

**Files:**
- Modify: `src/satec/models/train.py`
- Test: `tests/models/test_train.py`

**Interfaces:**
- Produces:
  - `train_random_forest(X, y, n_estimators=300, max_depth=None) -> RandomForestClassifier`
  - `train_gradient_boosting(X, y, max_iter=300) -> HistGradientBoostingClassifier`
  - Ambos exponen `.predict(X)` y `.predict_proba(X)[:, 1]`.

- [ ] **Step 1: Añadir tests que fallan** (append a `tests/models/test_train.py`)

```python
from satec.models.train import train_random_forest, train_gradient_boosting


def test_random_forest_aprende_patron_separable():
    X, y = _xy()
    clf = train_random_forest(X, y, n_estimators=50)
    assert (clf.predict(X) == y).mean() > 0.9
    assert clf.predict_proba(X).shape[1] == 2


def test_gradient_boosting_aprende_patron_separable():
    X, y = _xy()
    clf = train_gradient_boosting(X, y, max_iter=50)
    assert (clf.predict(X) == y).mean() > 0.9
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/models/test_train.py -v`
Expected: FAIL con `ImportError: cannot import name 'train_random_forest'`.

- [ ] **Step 3: Implementar** (añadir a `src/satec/models/train.py`)

```python
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier)


def train_random_forest(X, y, n_estimators=300, max_depth=None):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", random_state=42, n_jobs=-1)
    return clf.fit(X, y)


def train_gradient_boosting(X, y, max_iter=300):
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, class_weight="balanced",
        learning_rate=0.1, random_state=42)
    return clf.fit(X, y)
```

> **Si sklearn < 1.5** (Task 1 lo detecta): `HistGradientBoostingClassifier` no acepta `class_weight`. Sustituir su cuerpo por:
> ```python
> import numpy as np
> from sklearn.utils.class_weight import compute_sample_weight
> def train_gradient_boosting(X, y, max_iter=300):
>     clf = HistGradientBoostingClassifier(max_iter=max_iter,
>                                          learning_rate=0.1, random_state=42)
>     sw = compute_sample_weight("balanced", y)
>     return clf.fit(X, y, sample_weight=sw)
> ```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/models/test_train.py -v`
Expected: PASS (los tres tests).

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/train.py tests/models/test_train.py
git commit -m "feat(models): añade entrenadores Random Forest y Gradient Boosting"
```

### Task 4: Selector de umbral óptimo

**Files:**
- Create: `src/satec/models/thresholds.py`
- Test: `tests/models/test_thresholds.py`

**Interfaces:**
- Produces: `best_threshold(y_true, y_score, beta=1.0) -> (t_estrella: float, f_estrella: float)`. Devuelve `(0.5, 0.0)` si `y_true` no tiene positivos (sin señal para optimizar).

- [ ] **Step 1: Escribir el test que falla**

```python
import numpy as np
from satec.models.thresholds import best_threshold


def test_best_threshold_separa_clases():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    t, f = best_threshold(y_true, y_score, beta=1.0)
    assert 0.2 < t <= 0.8     # cualquier umbral en ese rango da F1=1
    assert abs(f - 1.0) < 1e-9


def test_best_threshold_sin_positivos_devuelve_default():
    t, f = best_threshold([0, 0, 0], [0.1, 0.2, 0.3])
    assert t == 0.5 and f == 0.0
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/models/test_thresholds.py -v`
Expected: FAIL con `ModuleNotFoundError: satec.models.thresholds`.

- [ ] **Step 3: Implementar**

```python
"""Selección de umbral de decisión que maximiza F-beta (sin tocar el test)."""
import numpy as np
from sklearn.metrics import fbeta_score


def best_threshold(y_true, y_score, beta=1.0):
    """Devuelve (t*, f*): el umbral que maximiza F-beta sobre y_score.

    Si no hay positivos en y_true no hay señal para optimizar y se devuelve el
    umbral neutro 0.5.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0:
        return 0.5, 0.0
    candidatos = np.unique(np.concatenate([[0.0], np.sort(y_score), [1.0]]))
    best_t, best_f = 0.5, -1.0
    for t in candidatos:
        pred = (y_score >= t).astype(int)
        f = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, float(best_f)
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/models/test_thresholds.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/thresholds.py tests/models/test_thresholds.py
git commit -m "feat(models): selector de umbral óptimo por F-beta"
```

### Task 5: Validación de origen móvil

**Files:**
- Create: `src/satec/models/rolling_origin.py`
- Test: `tests/models/test_rolling_origin.py`

**Interfaces:**
- Consumes: `feature_matrix` (features_matrix.py), `best_threshold` (Task 4), entrenadores (Task 3 + existentes), `evaluate_predictions` (Task 2).
- Produces:
  - `pooled_predictions(df, build_fn, score_fn, test_years, val_min_pos=5, beta=1.0) -> dict` con claves `y_true`, `y_score`, `y_pred` (np.ndarray concatenados sobre todos los años de prueba) y `umbrales` (lista de (anio, t*)).
  - `rolling_evaluation(df, test_years=range(2016, 2025), nn_epochs=60, beta=1.0) -> pandas.DataFrame` con una fila por modelo y columnas: `modelo, accuracy, precision, recall, f1, f2, especificidad, roc_auc, pr_auc, brier, umbral, tn, fp, fn, tp`.
- `build_fn(Xtr, ytr)` devuelve un objeto modelo; `score_fn(modelo, X)` devuelve `np.ndarray` de probabilidades de la clase 1. Esto unifica sklearn y la RN.

- [ ] **Step 1: Escribir el test que falla**

```python
import numpy as np
import pandas as pd
from satec.models.features_matrix import FEATURE_COLS
from satec.models.rolling_origin import pooled_predictions, rolling_evaluation


def _ds(n_per_year=120):
    rng = np.random.RandomState(0)
    rows = []
    for anio in range(2012, 2020):
        for _ in range(n_per_year):
            feats = {c: rng.rand() for c in FEATURE_COLS}
            # señal: brote depende de 'roll_mean8'
            brote = int(feats["roll_mean8"] > 0.8)
            rows.append({**feats, "departamento": "X", "provincia": "P",
                         "ubigeo_prov": "0201", "anio": anio, "semana": 5,
                         "q1": 0.0, "q2": 0.0, "q3": 1.0, "casos": 0.0,
                         "brote": brote})
    return pd.DataFrame(rows)


def test_pooled_predictions_respeta_anios_de_prueba():
    from sklearn.tree import DecisionTreeClassifier
    df = _ds()
    build = lambda X, y: DecisionTreeClassifier(random_state=0).fit(X, y)
    score = lambda m, X: m.predict_proba(X)[:, 1]
    out = pooled_predictions(df, build, score, test_years=range(2015, 2020),
                             val_min_pos=1)
    # 5 años de prueba × 120 filas
    assert len(out["y_true"]) == 5 * 120
    assert set(np.unique(out["y_pred"])) <= {0, 1}


def test_rolling_evaluation_incluye_seis_modelos():
    df = _ds()
    res = rolling_evaluation(df, test_years=range(2015, 2020), nn_epochs=2)
    assert {"red_neuronal", "random_forest", "gradient_boosting",
            "arbol_poda8", "arbol_sin_poda", "baseline_persistencia"} <= set(res["modelo"])
    for col in ["f1", "f2", "especificidad", "pr_auc", "umbral"]:
        assert col in res.columns
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/models/test_rolling_origin.py -v`
Expected: FAIL con `ModuleNotFoundError: satec.models.rolling_origin`.

- [ ] **Step 3: Implementar**

```python
"""Validación de origen móvil: entrena por corte temporal, elige umbral en
validación (sin tocar el año de prueba) y agrupa las predicciones fuera de
muestra de todos los años de prueba."""
import numpy as np
import pandas as pd

from satec.models.features_matrix import feature_matrix
from satec.models.metrics import evaluate_predictions
from satec.models.thresholds import best_threshold
from satec.models.train import (train_decision_tree, train_random_forest,
                                train_gradient_boosting, train_neural_net,
                                nn_predict_proba)


def _val_window(df, year, val_min_pos):
    """Años [year-1, year-2, ...] (estrictamente previos) hasta reunir
    val_min_pos brotes; devuelve el sub-DataFrame de validación."""
    val = df[df["anio"] == year - 1]
    k = 2
    while int(val["brote"].sum()) < val_min_pos and (year - k) >= df["anio"].min():
        val = df[(df["anio"] >= year - k) & (df["anio"] <= year - 1)]
        k += 1
    return val


def pooled_predictions(df, build_fn, score_fn, test_years,
                       val_min_pos=5, beta=1.0):
    yt, ys, yp, umbrales = [], [], [], []
    for year in test_years:
        train_inner = df[df["anio"] <= year - 2]
        val = _val_window(df, year, val_min_pos)
        test = df[df["anio"] == year]
        if len(train_inner) == 0 or len(test) == 0:
            continue
        Xtr, ytr = feature_matrix(train_inner)
        model = build_fn(Xtr, ytr)
        # umbral en validación; si no hay validación útil, 0.5
        if len(val) and int(val["brote"].sum()) > 0:
            Xv, yv = feature_matrix(val)
            t, _ = best_threshold(yv, score_fn(model, Xv), beta=beta)
        else:
            t = 0.5
        Xte, yte = feature_matrix(test)
        s = score_fn(model, Xte)
        yt.append(yte.to_numpy()); ys.append(s)
        yp.append((s >= t).astype(int)); umbrales.append((int(year), float(t)))
    return {"y_true": np.concatenate(yt), "y_score": np.concatenate(ys),
            "y_pred": np.concatenate(yp), "umbrales": umbrales}


def _row(modelo, pooled):
    m = evaluate_predictions(pooled["y_true"], pooled["y_pred"],
                             pooled["y_score"])
    ts = [t for _, t in pooled["umbrales"]]
    m.update({"modelo": modelo,
              "umbral": float(np.mean(ts)) if ts else 0.5})
    return m


def _nn_build(Xtr, ytr):
    return train_neural_net(Xtr, ytr, epochs=_nn_build.epochs)


def rolling_evaluation(df, test_years=range(2016, 2025), nn_epochs=60, beta=1.0):
    test_years = list(test_years)
    rows = []

    especificaciones = {
        "arbol_sin_poda": (lambda X, y: train_decision_tree(X, y, max_depth=None),
                           lambda m, X: m.predict_proba(X)[:, 1]),
        "arbol_poda8": (lambda X, y: train_decision_tree(X, y, max_depth=8),
                        lambda m, X: m.predict_proba(X)[:, 1]),
        "random_forest": (lambda X, y: train_random_forest(X, y),
                          lambda m, X: m.predict_proba(X)[:, 1]),
        "gradient_boosting": (lambda X, y: train_gradient_boosting(X, y),
                              lambda m, X: m.predict_proba(X)[:, 1]),
    }
    for nombre, (build, score) in especificaciones.items():
        pooled = pooled_predictions(df, build, score, test_years, beta=beta)
        rows.append(_row(nombre, pooled))

    # Red neuronal: build devuelve (model, norm); score usa nn_predict_proba.
    _nn_build.epochs = nn_epochs
    pooled_rn = pooled_predictions(
        df, _nn_build,
        lambda mn, X: nn_predict_proba(mn[0], X, mn[1]),
        test_years, beta=beta)
    rows.append(_row("red_neuronal", pooled_rn))

    # Baseline canal endémico: marca brote si casos > q3 (umbral fijo, sin score).
    yt, yp = [], []
    for year in test_years:
        test = df[df["anio"] == year]
        if len(test) == 0:
            continue
        _, yte = feature_matrix(test)
        yt.append(yte.to_numpy())
        yp.append((test["casos"] > test["q3"]).astype(int).to_numpy())
    bp = {"y_true": np.concatenate(yt), "y_pred": np.concatenate(yp),
          "y_score": None, "umbrales": []}
    m = evaluate_predictions(bp["y_true"], bp["y_pred"])
    m.update({"modelo": "baseline_persistencia", "umbral": float("nan")})
    rows.append(m)

    cols = ["modelo", "accuracy", "precision", "recall", "f1", "f2",
            "especificidad", "roc_auc", "pr_auc", "brier", "umbral",
            "tn", "fp", "fn", "tp"]
    return pd.DataFrame(rows)[cols]
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/models/test_rolling_origin.py -v`
Expected: PASS (los dos tests).

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/rolling_origin.py tests/models/test_rolling_origin.py
git commit -m "feat(models): validación de origen móvil con umbral en validación"
```

### Task 6: Orquestador de evaluación y CSV ampliado

**Files:**
- Modify: `src/satec/models/evaluate.py`
- Test: `tests/models/test_evaluate.py`

**Interfaces:**
- Consumes: `rolling_evaluation` (Task 5).
- Produces:
  - `run_evaluation(df, year_cutoff=2019, nn_epochs=60) -> DataFrame` (corte único; se conserva, ahora con 6 modelos y columnas `f2`, `especificidad`, `umbral`).
  - `main(repo=".", modo="rolling")` escribe `results/metricas_modelos.csv` (modo `"rolling"` por defecto) y `results/metricas_modelos_corte_unico.csv` (modo de robustez).

- [ ] **Step 1: Actualizar el test** (reemplazar `test_run_evaluation_devuelve_metricas_por_modelo` en `tests/models/test_evaluate.py`)

```python
def test_run_evaluation_incluye_seis_modelos_y_f2():
    res = run_evaluation(_dataset(), year_cutoff=2019, nn_epochs=3)
    modelos = set(res["modelo"])
    assert {"arbol_sin_poda", "arbol_poda8", "random_forest",
            "gradient_boosting", "red_neuronal",
            "baseline_persistencia"} <= modelos
    for col in ["recall", "pr_auc", "f2", "especificidad", "umbral"]:
        assert col in res.columns
    assert res["recall"].notna().all()
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/models/test_evaluate.py -v`
Expected: FAIL (faltan `random_forest`/`gradient_boosting`/`f2`).

- [ ] **Step 3: Implementar**

En `src/satec/models/evaluate.py`: importar los nuevos entrenadores, `best_threshold` y `rolling_evaluation`; extender `run_evaluation` para añadir RF, GB y, en la RN, elegir el umbral en una validación interna (último año del train). Reemplazar el bloque de la RN y añadir los ensembles:

```python
from satec.models.train import (train_decision_tree, train_random_forest,
                                train_gradient_boosting, train_neural_net,
                                nn_predict_proba)
from satec.models.thresholds import best_threshold
from satec.models.rolling_origin import rolling_evaluation
```

Dentro de `run_evaluation`, tras los árboles, añadir los ensembles:

```python
    ensembles = {
        "random_forest": train_random_forest(Xtr, ytr),
        "gradient_boosting": train_gradient_boosting(Xtr, ytr),
    }
    for nombre, clf in ensembles.items():
        score = clf.predict_proba(Xte)[:, 1]
        pred = clf.predict(Xte)
        train_acc = float((clf.predict(Xtr) == ytr.to_numpy()).mean())
        rows.append(_row(nombre, yte, pred, score, train_acc))
```

Reemplazar el bloque de la RN para usar umbral óptimo elegido en validación interna (último año de train):

```python
    model, norm = train_neural_net(Xtr, ytr, epochs=nn_epochs)
    score = nn_predict_proba(model, Xte, norm)
    val_year = int(train_df["anio"].max())
    val = train_df[train_df["anio"] == val_year]
    if len(val) and int(val["brote"].sum()) > 0:
        Xv, yv = feature_matrix(val)
        t_rn, _ = best_threshold(yv, nn_predict_proba(model, Xv, norm))
    else:
        t_rn = 0.5
    pred = (score >= t_rn).astype(int)
    train_score = nn_predict_proba(model, Xtr, norm)
    train_acc = float(((train_score >= t_rn).astype(int) == ytr.to_numpy()).mean())
    rows.append(_row("red_neuronal", yte, pred, score, train_acc))
```

Extender `_row` para incluir `"umbral"` (las filas de árbol/RF/GB usan su umbral por defecto 0.5; la RN y el baseline pasan el suyo). Forma mínima: añadir `m.setdefault("umbral", 0.5)` en `_row` y, para la RN, `rows[-1]["umbral"] = t_rn` tras el append. Actualizar la lista `cols` para incluir `"f2", "especificidad", "umbral"`.

Reescribir `main` para los dos modos:

```python
def main(repo: str = ".", modo: str = "rolling") -> None:
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    os.makedirs(os.path.join(repo, "results"), exist_ok=True)
    if modo == "rolling":
        res = rolling_evaluation(df, test_years=range(2016, 2025))
        out = os.path.join(repo, "results", "metricas_modelos.csv")
    else:
        res = run_evaluation(df, year_cutoff=2019)
        out = os.path.join(repo, "results", "metricas_modelos_corte_unico.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 180)
    print(res.to_string(index=False))
    print(f"\n[OK] -> {out}")
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/models/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 5: Generar los CSV reales y revisar el F1**

Run:
```bash
python -m satec.models.evaluate            # modo rolling -> metricas_modelos.csv
python -c "from satec.models.evaluate import main; main(modo='corte_unico')"
```
Expected: imprime las tablas; el `metricas_modelos.csv` (origen móvil) muestra un F1 de la RN/RF claramente superior al 0,39 del corte único. **Checkpoint humano:** anotar los números para el paper.

- [ ] **Step 6: Commit**

```bash
git add src/satec/models/evaluate.py tests/models/test_evaluate.py results/metricas_modelos.csv results/metricas_modelos_corte_unico.csv
git commit -m "feat(models): evaluación de 6 modelos con origen móvil y umbral óptimo"
```

---

## FASE 2 — Figuras de calidad de revista

### Task 7: Figura de la arquitectura de la red neuronal (nota p115)

**Files:**
- Create: `src/satec/models/figure_architecture.py`
- Test: `tests/models/test_figure_architecture.py`

**Interfaces:**
- Produces: `plot_architecture(out_path)` — guarda un PNG con el diagrama 24→32→32→1 anotado (entrada, capas ReLU, salida sigmoide, normalización min-max, pérdida).

- [ ] **Step 1: Escribir el test que falla**

```python
import os
from satec.models.figure_architecture import plot_architecture


def test_plot_architecture_crea_png(tmp_path):
    out = tmp_path / "fig_arquitectura.png"
    plot_architecture(str(out))
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/models/test_figure_architecture.py -v`
Expected: FAIL con `ModuleNotFoundError`.

- [ ] **Step 3: Implementar**

```python
"""Figura esquemática de la arquitectura de la red neuronal (24-32-32-1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from satec.models.paper_style import apply_style, AZUL, NARANJA, VERDE, GRIS


def _caja(ax, x, y, w, h, texto, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.2, edgecolor=color,
                                facecolor=color + "22"))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=10)


def plot_architecture(out_path):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    capas = [("Entrada\n24 variables\n(normalización\nmin-max)", AZUL),
             ("Oculta 1\n32 · ReLU", NARANJA),
             ("Oculta 2\n32 · ReLU", NARANJA),
             ("Salida\n1 · sigmoide\nP(brote)", VERDE)]
    x = 0.3
    centros = []
    for texto, color in capas:
        _caja(ax, x, 0.9, 1.5, 1.4, texto, color)
        centros.append(x + 1.5)
        x += 2.3
    for cx in centros[:-1]:
        ax.add_patch(FancyArrowPatch((cx, 1.6), (cx + 0.8, 1.6),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=GRIS, linewidth=1.2))
    ax.text(x + 0.1, 1.6,
            "Pérdida:\nentropía cruzada\nbinaria · Adam\nclass_weight\nbalanceado",
            ha="left", va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, x + 1.8); ax.set_ylim(0.4, 2.8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(repo: str = ".") -> None:
    import os
    plot_architecture(os.path.join(repo, "results", "fig_arquitectura.png"))
    print("[OK] fig_arquitectura.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Ejecutar y verificar que pasa**

Run: `pytest tests/models/test_figure_architecture.py -v`
Expected: PASS.

- [ ] **Step 5: Generar la figura y commit**

```bash
python -m satec.models.figure_architecture
git add src/satec/models/figure_architecture.py tests/models/test_figure_architecture.py results/fig_arquitectura.png
git commit -m "feat(figuras): diagrama de la arquitectura de la red neuronal"
```

### Task 8: Nombres y colores de RF/GB en el estilo

**Files:**
- Modify: `src/satec/models/paper_style.py`
- Test: `tests/models/test_figures.py` (añadir aserción de nombres)

**Interfaces:**
- Produces: `MODELOS` incluye `"random_forest"` y `"gradient_boosting"`; `nice_model` los traduce.

- [ ] **Step 1: Añadir test que falla** (append a `tests/models/test_figures.py`)

```python
from satec.models.paper_style import nice_model

def test_nombres_de_ensembles():
    assert nice_model("random_forest") == "Random Forest"
    assert nice_model("gradient_boosting") == "Gradient Boosting"
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/models/test_figures.py::test_nombres_de_ensembles -v`
Expected: FAIL (devuelve la clave cruda).

- [ ] **Step 3: Implementar** (en `MODELOS` de `paper_style.py`)

```python
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
```

- [ ] **Step 4: Verificar que pasa**

Run: `pytest tests/models/test_figures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/paper_style.py tests/models/test_figures.py
git commit -m "feat(figuras): nombres legibles de Random Forest y Gradient Boosting"
```

### Task 9: Barras con etiquetas numéricas y 6 modelos (notas p124, p156)

**Files:**
- Modify: `src/satec/models/figures.py`, `src/satec/models/figures_extra.py`
- Test: `tests/models/test_figures.py`

**Interfaces:**
- Consumes: `results/metricas_modelos.csv` (Task 6).
- Produces: `fig_metricas.png`, `fig_brecha.png`, `fig_f1.png` con etiquetas numéricas; `ORDEN` incluye RF y GB.

- [ ] **Step 1: Añadir test** (append a `tests/models/test_figures.py`) — verifica que las funciones aceptan los 6 modelos y generan archivo

```python
import os
import pandas as pd
from satec.models.figures import plot_metrics_bar

def test_metrics_bar_con_seis_modelos(tmp_path):
    filas = []
    for m in ["red_neuronal", "random_forest", "gradient_boosting",
              "arbol_poda8", "arbol_sin_poda", "baseline_persistencia"]:
        filas.append({"modelo": m, "recall": 0.5, "pr_auc": 0.5, "f1": 0.5})
    df = pd.DataFrame(filas)
    out = tmp_path / "m.png"
    plot_metrics_bar(df, str(out))
    assert out.exists()
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/models/test_figures.py::test_metrics_bar_con_seis_modelos -v`
Expected: FAIL (KeyError por reindex con modelos ausentes en `ORDEN`).

- [ ] **Step 3: Implementar**

En `figures.py`: ampliar `ORDEN`:

```python
ORDEN = ["red_neuronal", "random_forest", "gradient_boosting",
         "arbol_poda8", "arbol_sin_poda", "baseline_persistencia"]
```

En `plot_metrics_bar`, hacer el reindex robusto y añadir etiquetas numéricas sobre cada barra:

```python
    sub = res_df.set_index("modelo").reindex([m for m in ORDEN
                                              if m in set(res_df["modelo"])])
```
Tras dibujar cada serie de barras `bars = ax.bar(...)`, añadir:
```python
        for rect, val in zip(bars, sub[m].to_numpy(dtype=float)):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
```
Aplicar el mismo patrón de etiquetas en `plot_generalization_gap` (ya las tiene) y en `figures_extra.plot_f1`, y ampliar la lista `modelos` de `plot_f1` con `"random_forest"` y `"gradient_boosting"`.

- [ ] **Step 4: Verificar que pasa**

Run: `pytest tests/models/test_figures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/satec/models/figures.py src/satec/models/figures_extra.py tests/models/test_figures.py
git commit -m "feat(figuras): etiquetas numéricas y 6 modelos en las barras"
```

### Task 10: Matrices de confusión RN + RF con porcentajes (notas p111, p116, p118–p122)

**Files:**
- Modify: `src/satec/models/figures_extra.py`
- Test: `tests/models/test_figures.py`

**Interfaces:**
- Produces: `plot_confusions(res, out_path)` dibuja **red_neuronal** y **random_forest**, con conteo y porcentaje por celda y etiquetas VP/FP/FN/VN.

- [ ] **Step 1: Añadir test** (append a `tests/models/test_figures.py`)

```python
from satec.models.figures_extra import plot_confusions

def test_confusiones_rn_y_rf(tmp_path):
    filas = []
    for m in ["red_neuronal", "random_forest"]:
        filas.append({"modelo": m, "tn": 100, "fp": 10, "fn": 5, "tp": 20})
    df = pd.DataFrame(filas)
    out = tmp_path / "c.png"
    plot_confusions(df, str(out))
    assert out.exists()
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/models/test_figures.py::test_confusiones_rn_y_rf -v`
Expected: FAIL (la versión actual usa `arbol_poda8`; con este df busca `red_neuronal` y `random_forest`, pero aún no añade porcentajes ni acepta el cambio — ajustar para que falle por el modelo ausente o por aserción del archivo).

> Nota: el test pasará en cuanto `plot_confusions` lea `random_forest`. El valor de la tarea está en los porcentajes/etiquetas; verificarlos en el Step 5 (revisión visual).

- [ ] **Step 3: Implementar**

En `figures_extra.plot_confusions`, cambiar `modelos = ["red_neuronal", "arbol_poda8"]` por `["red_neuronal", "random_forest"]` y anotar cada celda con conteo + porcentaje sobre el total, además de la etiqueta de cuadrante:

```python
        total = cm.sum()
        etiquetas = [["VN", "FP"], ["FN", "VP"]]
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / total
                ax.text(j, i, f"{etiquetas[i][j]}\n{cm[i, j]:,}\n{pct:.2f}%",
                        ha="center", va="center",
                        color="white" if norm[i, j] > 0.55 else "#222222",
                        fontsize=11, fontweight="bold")
```

- [ ] **Step 4: Verificar que pasa**

Run: `pytest tests/models/test_figures.py -v`
Expected: PASS.

- [ ] **Step 5: Regenerar todas las figuras de métricas y revisar**

Run:
```bash
python -m satec.models.figures
python -m satec.models.figures_extra
```
Expected: `fig_metricas.png`, `fig_brecha.png`, `fig_f1.png`, `fig_confusion.png` actualizadas; las matrices muestran VP/FP/FN/VN con % (revisión visual).

- [ ] **Step 6: Commit**

```bash
git add src/satec/models/figures_extra.py tests/models/test_figures.py results/fig_*.png
git commit -m "feat(figuras): matrices de confusión RN y RF con conteo y porcentaje"
```

### Task 11: Calibración sobre el pool de origen móvil (nota p126)

**Files:**
- Modify: `src/satec/models/run_interpret.py`
- Test: ninguno nuevo (script de generación); verificación visual.

**Interfaces:**
- Consumes: `pooled_predictions` (Task 5), `calibration_points`, `plot_calibration` (interpret.py).
- Produces: `fig_calibracion.png` (RN y RF) e `fig_importancia.png` con muchos más positivos → curva estable.

- [ ] **Step 1: Reescribir `main` de `run_interpret.py`**

Sustituir el cálculo por uno basado en el pool de origen móvil (RN y RF), reutilizando `pooled_predictions`:

```python
import os
import pandas as pd
from satec.models.rolling_origin import pooled_predictions, _nn_build
from satec.models.train import (train_random_forest, train_neural_net,
                                nn_predict_proba)
from satec.models.features_matrix import feature_matrix
from satec.models.interpret import (permutation_importance_ap,
                                    calibration_points, plot_importance,
                                    plot_calibration)
from satec.models.paper_style import nice_model


def main(repo: str = ".") -> None:
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    results = os.path.join(repo, "results"); os.makedirs(results, exist_ok=True)
    years = range(2016, 2025)

    rf_pool = pooled_predictions(
        df, lambda X, y: train_random_forest(X, y),
        lambda m, X: m.predict_proba(X)[:, 1], years)
    _nn_build.epochs = 60
    rn_pool = pooled_predictions(
        df, _nn_build, lambda mn, X: nn_predict_proba(mn[0], X, mn[1]), years)

    curves = {
        nice_model("red_neuronal"): calibration_points(
            rn_pool["y_true"], rn_pool["y_score"], n_bins=10),
        nice_model("random_forest"): calibration_points(
            rf_pool["y_true"], rf_pool["y_score"], n_bins=10),
    }
    plot_calibration(curves, os.path.join(results, "fig_calibracion.png"))

    # Importancia por permutación de la RN entrenada con todo el histórico.
    Xall, yall = feature_matrix(df[df["anio"] <= 2024])
    model, norm = train_neural_net(Xall, yall, epochs=60)
    imp = permutation_importance_ap(
        lambda X: nn_predict_proba(model, X, norm), Xall, yall, n_repeats=5)
    imp.to_csv(os.path.join(results, "importancia_variables.csv"), index=False)
    plot_importance(imp, os.path.join(results, "fig_importancia.png"))
    print("[OK] fig_calibracion.png · fig_importancia.png")
```

- [ ] **Step 2: Generar y revisar**

Run: `python -m satec.models.run_interpret`
Expected: `fig_calibracion.png` con curvas RN/RF más suaves; revisión visual de que están cerca de la diagonal.

- [ ] **Step 3: Commit**

```bash
git add src/satec/models/run_interpret.py results/fig_calibracion.png results/fig_importancia.png results/importancia_variables.csv
git commit -m "fix(figuras): calibración estable sobre el pool de origen móvil"
```

---

## FASE 3 — Web (consistencia con el paper)

### Task 12: Re-exportar modelos a la web

**Files:**
- Modify: `src/satec/web_export/export_models.py`
- Test: `tests/web_export/test_export_tree.py` (existente, debe seguir verde)

**Interfaces:**
- Produces: `web/public/models/ad.json`, `web/public/models/rn/*`, `web/public/models/norm.json` regenerados con los modelos actuales.

- [ ] **Step 1: Ejecutar la exportación actual**

Run: `python -m satec.web_export.export_models`
Expected: `[OK] modelos exportados a web/public/models`.

- [ ] **Step 2 (opcional, confirmar con el usuario): añadir Random Forest a la web**

Si se incluye RF: serializar cada estimador con `tree_to_dict` en una lista `web/public/models/rf.json`. Añadir a `export_all`:

```python
    from satec.models.train import train_random_forest
    rf = train_random_forest(X, y)
    rf_json = {"n_estimators": len(rf.estimators_),
               "trees": [tree_to_dict(e, FEATURE_COLS, CLASS_LABELS)
                         for e in rf.estimators_]}
    with open(os.path.join(out_dir, "rf.json"), "w", encoding="utf-8") as f:
        json.dump(rf_json, f)
```
> El consumo de `rf.json` en `web/public/app.js` (promedio de las hojas de los árboles) queda fuera de este plan salvo que el usuario lo pida; documentarlo como pendiente.

- [ ] **Step 3: Verificar que los tests de export siguen verdes**

Run: `pytest tests/web_export -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/satec/web_export/export_models.py web/public/models
git commit -m "chore(web): re-exporta los modelos actualizados para el mapa de riesgo"
```

---

## FASE 4 — Notebooks Colab/Jupyter

### Task 13: Notebook de Random Forest + Gradient Boosting y actualización de 01/02

**Files:**
- Modify: `notebooks/build_notebooks.py`
- Create (generado): `notebooks/03_random_forest_train_test.ipynb`

**Interfaces:**
- Produces: tres notebooks regenerables con `python notebooks/build_notebooks.py`.

- [ ] **Step 1: Corregir el comentario de prevalencia en los bloques comunes**

En `build_notebooks.py`, el docstring de `evaluar` dice "brotes ~8%". Cambiarlo por el dato real y causal:
```python
    """Metricas adecuadas a eventos raros (la prevalencia de brotes cae del
    ~10% (<=2019) al ~0.3% en 2020-2024 por la subnotificacion pandemica)."""
```

- [ ] **Step 2: Añadir `build_rf()`** en `build_notebooks.py` (análogo a `build_arbol`), con celdas que entrenan `RandomForestClassifier` y `HistGradientBoostingClassifier(class_weight="balanced")`, evalúan con `evaluar`, y dibujan importancia y matriz de confusión. Reutilizar `FEATURES_SRC`, `SPLIT_SRC`, `METRICS_SRC`, `CONF_PLOT_SRC`, `BASELINE_SRC`. Bloque de entrenamiento:

```python
        code('from sklearn.ensemble import (RandomForestClassifier,\n'
             '                              HistGradientBoostingClassifier)\n\n'
             'rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",\n'
             '                            random_state=42, n_jobs=-1).fit(Xtr, ytr)\n'
             'gb = HistGradientBoostingClassifier(max_iter=300,\n'
             '          class_weight="balanced", random_state=42).fit(Xtr, ytr)'),
```

- [ ] **Step 3: Añadir una sección de origen móvil** (celda markdown + código) a los tres notebooks, que reproduzca el esquema train≤Y-2 / val=Y-1 / test=Y para el modelo del cuaderno y reporte el F1 con umbral óptimo. Código de la celda (autocontenido, sin depender del paquete `satec`):

```python
        code('from sklearn.metrics import fbeta_score, average_precision_score\n'
             'import numpy as np\n'
             'def best_t(yv, sv, beta=1.0):\n'
             '    if int(np.sum(yv))==0: return 0.5\n'
             '    cand=np.unique(np.concatenate([[0.0],np.sort(sv),[1.0]]))\n'
             '    return float(max(cand, key=lambda t: fbeta_score(yv,(sv>=t).astype(int),beta=beta,zero_division=0)))\n'
             'yt=[];yp=[];ys=[]\n'
             'for Y in range(2016,2025):\n'
             '    tr=df[df.anio<=Y-2]; va=df[df.anio==Y-1]; te=df[df.anio==Y]\n'
             '    if len(tr)==0 or len(te)==0: continue\n'
             '    Xtr2,ytr2=feature_matrix(tr); clf2=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1).fit(Xtr2,ytr2)\n'
             '    Xv,yv=feature_matrix(va); t=best_t(yv.to_numpy(),clf2.predict_proba(Xv)[:,1]) if len(va) else 0.5\n'
             '    Xte2,yte2=feature_matrix(te); s=clf2.predict_proba(Xte2)[:,1]\n'
             '    yt+=list(yte2);ys+=list(s);yp+=list((s>=t).astype(int))\n'
             'print("Origen movil — RF:", evaluar(np.array(yt),np.array(yp),np.array(ys)))'),
```

- [ ] **Step 4: Regenerar los notebooks**

Run: `python notebooks/build_notebooks.py`
Expected: `[OK]` para los tres `.ipynb` (01, 02, 03).

- [ ] **Step 5: Commit**

```bash
git add notebooks/build_notebooks.py notebooks/01_arbol_decision_train_test.ipynb notebooks/02_red_neuronal_train_test.ipynb notebooks/03_random_forest_train_test.ipynb
git commit -m "docs(notebooks): cuaderno de Random Forest/GB y sección de origen móvil"
```

---

## FASE 5 — Artículo (rigor de revista)

### Task 14: Validador de linealidad de citas (notas p113, p12/p14/p15)

**Files:**
- Create: `paper/cite_lint.py`, `tests/paper/__init__.py`, `tests/paper/test_cite_lint.py`

**Interfaces:**
- Produces:
  - `parse_citations(md_text) -> list[int]` — números de cita en orden de aparición en el cuerpo (excluye la sección de Referencias).
  - `check_linearity(md_text) -> list[str]` — lista de problemas (cita fuera de orden de primera aparición, hueco, o cita sin referencia). Lista vacía ⇒ correcto.

- [ ] **Step 1: Escribir los tests que fallan**

```python
from paper.cite_lint import parse_citations, check_linearity

CUERPO_OK = "Texto [1], luego [2] y [3].\n## Referencias\n[1] A\n[2] B\n[3] C\n"
CUERPO_MAL = "Texto [1] y luego [17]. Más [3].\n## Referencias\n[1] A\n"

def test_parse_citations_orden_de_aparicion():
    assert parse_citations(CUERPO_OK) == [1, 2, 3]

def test_linearidad_correcta_no_reporta():
    assert check_linearity(CUERPO_OK) == []

def test_linearidad_detecta_salto():
    problemas = check_linearity(CUERPO_MAL)
    assert any("17" in p for p in problemas)
```

- [ ] **Step 2: Ejecutar y verificar que falla**

Run: `pytest tests/paper/test_cite_lint.py -v`
Expected: FAIL con `ModuleNotFoundError: paper.cite_lint`.

> Si falla por import de `paper`, crear `paper/__init__.py` vacío.

- [ ] **Step 3: Implementar `paper/cite_lint.py`**

```python
"""Verifica la linealidad de citas IEEE/ACM: primera aparición en orden 1..N."""
import re
import sys

_CITE = re.compile(r"\[(\d+)\]")


def _split_body(md_text):
    m = re.search(r"^##\s+Referencias", md_text, flags=re.MULTILINE)
    return md_text[:m.start()] if m else md_text


def parse_citations(md_text):
    body = _split_body(md_text)
    vistos, orden = set(), []
    for n in (int(x) for x in _CITE.findall(body)):
        if n not in vistos:
            vistos.add(n); orden.append(n)
    return orden


def check_linearity(md_text):
    orden = parse_citations(md_text)
    problemas = []
    for esperado, real in enumerate(orden, start=1):
        if real != esperado:
            problemas.append(
                f"Cita [{real}] aparece donde se esperaba [{esperado}] "
                f"(orden de primera aparición roto).")
            break
    return problemas


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "paper/SATEC_articulo_ACM.md"
    with open(path, encoding="utf-8") as f:
        problemas = check_linearity(f.read())
    if problemas:
        print("\n".join(problemas)); sys.exit(1)
    print("[OK] Citas en orden lineal de aparición.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verificar que pasa**

Run: `pytest tests/paper/test_cite_lint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add paper/cite_lint.py paper/__init__.py tests/paper
git commit -m "test(paper): validador de linealidad de citas IEEE/ACM"
```

### Task 15: Renumerar referencias en orden de aparición

**Files:**
- Modify: `paper/SATEC_articulo_ACM.md`

- [ ] **Step 1: Detectar el orden actual**

Run: `python paper/cite_lint.py`
Expected: FALLA, lista la primera cita fuera de orden (hoy aparecen [1],[17] antes que [3]…). Anotar la secuencia que devuelve `parse_citations`.

- [ ] **Step 2: Construir el mapa viejo→nuevo** según la primera aparición en el cuerpo (Introducción → … → Conclusiones). Renumerar **toda** cita del cuerpo y reordenar la lista de la sección **Referencias** para que queden 1..N en ese mismo orden. Mantener intactos los datos bibliográficos; solo cambia el número.

- [ ] **Step 3: Verificar linealidad**

Run: `python paper/cite_lint.py`
Expected: `[OK] Citas en orden lineal de aparición.`

- [ ] **Step 4: Commit**

```bash
git add paper/SATEC_articulo_ACM.md
git commit -m "docs(paper): renumera referencias en orden de aparición (IEEE/ACM)"
```

### Task 16: Métodos — origen móvil, RF/GB y Tabla 1 (notas p111, p116)

**Files:**
- Modify: `paper/SATEC_articulo_ACM.md`

- [ ] **Step 1: Reescribir §2.7 «Validación y métricas»** para describir la **validación de origen móvil** (train ≤ Y-2, validación = Y-1 con ampliación si faltan positivos, prueba = Y, para Y=2016…2024; métricas pooled), añadir **F2** (β=2, prioriza sensibilidad) y la selección de **umbral óptimo** en validación. Mantener las ecuaciones existentes y añadir la de F-beta:

```markdown
$$ F_\beta = (1+\beta^2)\,\frac{P \cdot R}{\beta^2 P + R} $$
```

- [ ] **Step 2: Añadir §2.x «Modelos de ensamble»** describiendo Random Forest (bagging de árboles con `class_weight` balanceado) y Gradient Boosting (boosting de histogramas), con 1–2 ecuaciones (voto promedio y aditivo) y su motivación en datos tabulares (referencias ya citadas sobre superioridad de árboles en tabular).

- [ ] **Step 3: Reemplazar la Tabla 1** con los valores del nuevo `results/metricas_modelos.csv` (origen móvil), 6 modelos y columnas Sensibilidad, Precisión, F1, **F2**, AUC-PR, AUC-ROC, Brier. Copiar las cifras exactas del CSV (Task 6, Step 5).

- [ ] **Step 4: Verificar que las cifras del texto coinciden con el CSV** (revisión cruzada manual contra `metricas_modelos.csv`).

- [ ] **Step 5: Commit**

```bash
git add paper/SATEC_articulo_ACM.md
git commit -m "docs(paper): métodos de origen móvil y ensembles; Tabla 1 ampliada"
```

### Task 17: Resultados — matriz de confusión explicada (notas p118–p122, p124)

**Files:**
- Modify: `paper/SATEC_articulo_ACM.md`

- [ ] **Step 1: Reescribir §3.2** con lenguaje claro y correcto. Definir, **para la unidad provincia-semana**: VP = provincia-semana correctamente alertada como brote inminente; FP = **falsa alarma** (se alertó y no hubo brote); FN = brote no anticipado; VN = correctamente no alertada. Incluir los conteos y **porcentajes** de la nueva matriz (RN y RF) y el **porcentaje de error** de cada modelo (FP/(FP+VN) y FN/(FN+VP)). Añadir explícitamente:

```markdown
Conviene precisar que SATEC **no diagnostica personas**: cada celda de la matriz
cuenta *provincias-semana*, no pacientes. Un falso positivo es una **falsa alarma
territorial** (se anticipó un brote que no se materializó), no un diagnóstico
clínico erróneo en un individuo.
```

- [ ] **Step 2: Conectar con la intuición clínica** sin afirmar algo falso: una frase que explique que, por analogía, una alta sensibilidad significa «no dejar pasar las zonas que sí entrarán en brote», que es la prioridad en alerta temprana (nota p119).

- [ ] **Step 3: Commit**

```bash
git add paper/SATEC_articulo_ACM.md
git commit -m "docs(paper): explica la matriz de confusión (provincia-semana) con %"
```

### Task 18: Comparación cuantitativa con la literatura (notas p151, p156)

**Files:**
- Modify: `paper/SATEC_articulo_ACM.md`

**Interfaces:**
- Requiere acceso web y **validación del usuario** de cada cifra antes de incluirla.

- [ ] **Step 1: Buscar cifras** de desempeño (AUC/precisión/sensibilidad) en estudios afines de ML en enfermedades vectoriales con vigilancia (dengue de Rufasto-Goche; Leishmania de Vadmal; nicho de *Lutzomyia* de Moo-Llanes; y 1–2 estudios recientes de alerta de brotes con ML). Usar búsqueda web y guardar título, métrica, valor y fuente.

- [ ] **Step 2: Checkpoint con el usuario** — presentar la tabla de cifras candidatas para que las valide/corrija. **No avanzar sin su visto bueno.**

- [ ] **Step 3: Añadir un párrafo + tabla comparativa** en §4 (Discusión) contrastando esas cifras con las de SATEC (AUC-PR, sensibilidad), señalando diferencias de tarea/enfermedad para una comparación honesta. Citar cada estudio (referencias ya numeradas en Task 15; añadir nuevas al final manteniendo linealidad si se incorporan estudios nuevos).

- [ ] **Step 4: Verificar linealidad de citas tras añadir referencias**

Run: `python paper/cite_lint.py`
Expected: `[OK]`.

- [ ] **Step 5: Commit**

```bash
git add paper/SATEC_articulo_ACM.md
git commit -m "docs(paper): comparación cuantitativa con la literatura afín"
```

### Task 19: Resúmenes, fuentes, limitaciones y agradecimientos (notas p112, p123, p147, p159)

**Files:**
- Modify: `paper/SATEC_articulo_ACM.md`

- [ ] **Step 1: Palabras clave dentro de los resúmenes** (p112): asegurar que las **Keywords** queden inmediatamente tras el Abstract (inglés) y añadir una línea **«Palabras clave:»** equivalente tras el Resumen (español).

- [ ] **Step 2: Fuente de datos** (p123): en §2.1 y en «Disponibilidad de datos», mencionar el enlace del MINSA y dejarlo como referencia citada (ya existe la entrada; verificar el `[N]`).

- [ ] **Step 3: Pies de figura con fuente** (p147): a cada figura propia añadir, en el caption, «Fuente: SATEC [N]» con la cita correspondiente (las figuras de captura de la web ya dicen «elaboración propia»; unificar el formato).

- [ ] **Step 4: Limitaciones** — quitar «se empleó una única partición temporal» (ya no aplica) y añadir una frase sobre el **efecto de la subnotificación pandémica 2020–2023** en la prevalencia observada.

- [ ] **Step 5: Agradecimientos** (p159): quitar la repetición de la institución (UNTELS) por estar ya en la afiliación de los autores.

- [ ] **Step 6: Commit**

```bash
git add paper/SATEC_articulo_ACM.md
git commit -m "docs(paper): keywords en resúmenes, fuentes de figuras, limitaciones y agradecimientos"
```

### Task 20: Formato del generador y regeneración del .docx (notas p114, p127)

**Files:**
- Modify: `paper/build_acm.py`

- [ ] **Step 1: Sin sangría** (p127): en `build_acm.py`, tras crear cada párrafo de estilo `Para`/`PostHeadPara`, fijar la sangría de primera línea a cero:

```python
from docx.shared import Pt
...
def _sin_sangria(p):
    p.paragraph_format.first_line_indent = Pt(0)
    return p
```
Aplicarlo en el parser donde se crean los párrafos de cuerpo (`add_runs(para(style), ...)`), envolviendo: `add_runs(_sin_sangria(para(style)), stripped)`.

- [ ] **Step 2: Verificar la linealidad de citas antes de compilar**

Run: `python paper/cite_lint.py`
Expected: `[OK]`.

- [ ] **Step 3: Regenerar el `.docx` y el `.pdf`**

Run: `python paper/build_acm.py`
Expected: `[OK] paper/SATEC_articulo_ACM.docx (N ecuaciones OMML)`. (El PDF/QA es paso manual de Word, según nota p114.)

- [ ] **Step 4: Revisión visual final del `.docx`** contra la tabla de las 19 notas del spec (§6): Tabla 1 con 6 modelos y F2; matrices con %; figura de arquitectura; citas en orden; keywords en ambos resúmenes; sin sangría; fuentes en figuras; agradecimientos corregidos.

- [ ] **Step 5: Commit**

```bash
git add paper/build_acm.py paper/SATEC_articulo_ACM.docx paper/SATEC_articulo_ACM.pdf
git commit -m "docs(paper): párrafos sin sangría y regenera el .docx ACM final"
```

---

## Self-Review (cobertura del spec)

- §5.1 Modelos RF/GB → Task 3, 13, 16. ✓
- §5.2 Origen móvil + umbral (incl. caso 2021 sin brotes) → Task 4, 5. ✓
- §5.3 Métricas (F2, especificidad) + CSV → Task 2, 6. ✓
- §5.4 Figuras (arquitectura, etiquetas, matrices RN+RF, calibración) → Task 7, 9, 10, 11. ✓
- §5.5 Texto del paper (métodos, matriz, literatura, limitaciones, agradecimientos) → Task 16, 17, 18, 19. ✓
- §5.6 Referencias/formato (linealidad, fuente, keywords, sin sangría) → Task 14, 15, 19, 20. ✓
- §5.7 Notebooks → Task 13. ✓
- §5.8 Web → Task 12. ✓
- §5.9 requirements → Task 1. ✓
- Las 19 notas del profesor (§6 del spec) quedan cubiertas por las tareas citadas en cada una.

**Verificación global final:** `pytest -q` en verde y revisión visual del `.docx` (Task 20, Step 4).
