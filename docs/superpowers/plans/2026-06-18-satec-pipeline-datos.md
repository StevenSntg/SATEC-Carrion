# Plan 1 — Pipeline de datos (núcleo MINSA) — SATEC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformar el CSV de vigilancia del MINSA (casos individuales de Carrión) en un dataset modelable `provincia × año × semana` con casos imputados, canal endémico causal, etiqueta de brote a 4 semanas y características temporales.

**Architecture:** Pipeline funcional en módulos pequeños y puros bajo `src/satec/`, cada uno con una responsabilidad y testeado de forma aislada con `pytest`. Un orquestador (`build_dataset.py`) encadena los módulos y persiste el dataset en `data/processed/`.

**Tech Stack:** Python 3.12 (`py -3.12`), pandas 3.0.3, numpy 1.26.4, pytest 9.0.3. (scikit-learn se usará en el Plan 3, no aquí.)

## Global Constraints

- Intérprete: **Python 3.12** invocado como `py -3.12` (Windows). El default del sistema es 3.14 — NO usarlo.
- Granularidad espacial: **provincia**. Granularidad temporal: **semana epidemiológica**.
- Semanas normalizadas a **1–52** (la semana 53, cuando exista, se reasigna a 52).
- Horizonte de predicción del target: **4 semanas** (ventana `[t+1, t+4]`).
- Canal endémico **causal**: para etiquetar el año `Y`, se calcula con los `n_ref` años **previos** (default 5, mínimo 3). Filas sin historia suficiente → target nulo y se descartan del dataset final.
- Definición de brote: en la ventana `[t+1,t+4]`, `casos > Q3(semana)` **y** `casos >= min_cases` (default 2).
- Selección de provincias endémicas: data-driven (`>= min_casos` históricos en `>= min_anios` años distintos).
- Commits: cuenta **stevensntg**, **sin** coautoría de Claude (omitir todo bloque `Co-Authored-By`).
- Datos crudos de entrada: `data/raw/carrion_minsa_2000_2024.csv` (copia del CSV del MINSA ya disponible).

---

### Task 1: Estructura del proyecto y arnés de pruebas

**Files:**
- Create: `src/satec/__init__.py`
- Create: `src/satec/config.py`
- Create: `src/satec/data/__init__.py`
- Create: `src/satec/features/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `data/raw/.gitkeep`

**Interfaces:**
- Produces: fixture `sample_raw` (pytest) → `pd.DataFrame` con la estructura cruda del MINSA para tests deterministas; `satec.config` con constantes `WEEKS`, `DEFAULT_PARAMS`.

- [ ] **Step 1: Crear `requirements.txt`**

```
pandas==3.0.3
numpy==1.26.4
pytest==9.0.3
pyarrow
```

- [ ] **Step 2: Crear `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = src
python_files = test_*.py
addopts = -q
```

- [ ] **Step 3: Crear los paquetes (archivos `__init__.py` vacíos)**

Crear `src/satec/__init__.py`, `src/satec/data/__init__.py`, `src/satec/features/__init__.py`, `tests/__init__.py` (todos con contenido vacío) y `data/raw/.gitkeep` (vacío).

- [ ] **Step 4: Crear `src/satec/config.py`**

```python
"""Constantes y parámetros por defecto del pipeline SATEC."""

WEEKS = list(range(1, 53))  # semanas epidemiológicas 1..52

DEFAULT_PARAMS = {
    "horizon": 4,        # semanas hacia adelante para el target
    "n_ref": 5,          # años de referencia para el canal endémico
    "min_ref": 3,        # mínimo de años de referencia exigidos
    "min_cases": 2,      # casos mínimos para considerar brote
    "endemic_min_casos": 10,   # casos históricos mínimos para provincia endémica
    "endemic_min_anios": 3,    # años distintos con casos para provincia endémica
}

RAW_COLUMNS = [
    "departamento", "provincia", "distrito", "localidad", "enfermedad",
    "ano", "semana", "diagnostic", "diresa", "ubigeo", "localcod",
    "edad", "tipo_edad", "sexo",
]
```

- [ ] **Step 5: Crear el fixture `sample_raw` en `tests/conftest.py`**

```python
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
```

- [ ] **Step 6: Verificar que pytest descubre la suite (vacía aún)**

Run: `py -3.12 -m pytest`
Expected: `no tests ran` (exit 0) — confirma que pytest y la estructura funcionan.

- [ ] **Step 7: Commit**

```bash
git add src tests requirements.txt pytest.ini data/raw/.gitkeep
git commit -m "chore(setup): estructura del proyecto y arnes de pruebas pytest"
```

---

### Task 2: Carga y limpieza del CSV del MINSA

**Files:**
- Create: `src/satec/data/load.py`
- Test: `tests/data/test_load.py`
- Create: `tests/data/__init__.py`

**Interfaces:**
- Consumes: `sample_raw` fixture.
- Produces:
  - `load_raw(path: str) -> pd.DataFrame`
  - `clean(df: pd.DataFrame) -> pd.DataFrame` → columnas exactas
    `["departamento","provincia","ubigeo","anio","semana","fase","edad_anios","sexo"]`;
    `anio:int`, `semana:int (1..52)`, `fase:str ("AGUDA"|"ERUPTIVA")`,
    `edad_anios:float`, `sexo:str ("M"|"F")`.

- [ ] **Step 1: Escribir el test que falla** (`tests/data/test_load.py`)

```python
import pandas as pd
from satec.data.load import clean


def test_clean_normaliza_semana_53_a_52(sample_raw):
    out = clean(sample_raw)
    assert out["semana"].max() <= 52
    assert (out["semana"] == 52).sum() == 1  # la fila de semana 53 pasa a 52


def test_clean_columnas_y_tipos(sample_raw):
    out = clean(sample_raw)
    assert list(out.columns) == [
        "departamento", "provincia", "ubigeo", "anio", "semana",
        "fase", "edad_anios", "sexo",
    ]
    assert out["anio"].dtype.kind == "i"
    assert set(out["fase"].unique()) <= {"AGUDA", "ERUPTIVA"}


def test_clean_convierte_edad_en_meses_a_anios(sample_raw):
    out = clean(sample_raw)
    fila_meses = out[(out["provincia"] == "HUARAL")]
    assert abs(float(fila_meses["edad_anios"].iloc[0]) - 1.0) < 1e-9
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest tests/data/test_load.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.data.load'`

- [ ] **Step 3: Implementar `src/satec/data/load.py`**

```python
"""Carga y limpieza del CSV crudo de vigilancia del MINSA."""
import pandas as pd

_FASE_MAP = {
    "ENFERMEDAD DE CARRION AGUDA": "AGUDA",
    "ENFERMEDAD DE CARRION ERUPTIVA": "ERUPTIVA",
}


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _edad_a_anios(edad, tipo_edad) -> float:
    t = str(tipo_edad).strip().upper()
    edad = float(edad)
    if t == "M":
        return edad / 12.0
    if t == "D":
        return edad / 365.0
    return edad


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fase"] = df["enfermedad"].map(_FASE_MAP)
    df = df.dropna(subset=["fase"])  # descarta filas fuera de las dos fases

    df["anio"] = df["ano"].astype(int)
    df["semana"] = df["semana"].astype(int).clip(upper=52)  # 53 -> 52

    df["edad_anios"] = df.apply(
        lambda r: _edad_a_anios(r["edad"], r["tipo_edad"]), axis=1
    )
    df["sexo"] = df["sexo"].astype(str).str.strip().str.upper()
    df["departamento"] = df["departamento"].astype(str).str.strip().str.upper()
    df["provincia"] = df["provincia"].astype(str).str.strip().str.upper()
    df["ubigeo"] = df["ubigeo"].astype(str).str.strip()

    cols = ["departamento", "provincia", "ubigeo", "anio", "semana",
            "fase", "edad_anios", "sexo"]
    return df[cols].reset_index(drop=True)
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest tests/data/test_load.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/data/load.py tests/data/__init__.py tests/data/test_load.py
git commit -m "feat(data): carga y limpieza del CSV del MINSA"
```

---

### Task 3: Agregación a provincia × año × semana

**Files:**
- Create: `src/satec/data/aggregate.py`
- Test: `tests/data/test_aggregate.py`

**Interfaces:**
- Consumes: salida de `clean()`.
- Produces: `aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame` → columnas
  `["departamento","provincia","anio","semana","casos"]`, una fila por combinación
  observada, `casos:int` = nº de registros.

- [ ] **Step 1: Escribir el test que falla** (`tests/data/test_aggregate.py`)

```python
from satec.data.load import clean
from satec.data.aggregate import aggregate_weekly


def test_aggregate_cuenta_casos(sample_raw):
    agg = aggregate_weekly(clean(sample_raw))
    fila = agg[(agg["provincia"] == "HUARAZ") & (agg["anio"] == 2010)
               & (agg["semana"] == 5)]
    assert int(fila["casos"].iloc[0]) == 2  # dos casos en HUARAZ 2010 s5


def test_aggregate_columnas(sample_raw):
    agg = aggregate_weekly(clean(sample_raw))
    assert list(agg.columns) == [
        "departamento", "provincia", "anio", "semana", "casos"]
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest tests/data/test_aggregate.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.data.aggregate'`

- [ ] **Step 3: Implementar `src/satec/data/aggregate.py`**

```python
"""Agregacion de casos individuales a conteos provincia x anio x semana."""
import pandas as pd

KEYS = ["departamento", "provincia", "anio", "semana"]


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(KEYS, as_index=False)
             .size()
             .rename(columns={"size": "casos"}))
    agg["casos"] = agg["casos"].astype(int)
    return agg[KEYS + ["casos"]]
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest tests/data/test_aggregate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/data/aggregate.py tests/data/test_aggregate.py
git commit -m "feat(data): agregacion semanal por provincia"
```

---

### Task 4: Rejilla completa e imputación de ceros

**Files:**
- Create: `src/satec/data/grid.py`
- Test: `tests/data/test_grid.py`

**Interfaces:**
- Consumes: salida de `aggregate_weekly()`.
- Produces:
  - `build_full_grid(agg: pd.DataFrame) -> pd.DataFrame` → producto cartesiano
    `(departamento,provincia) × anios_presentes × WEEKS`, columnas
    `["departamento","provincia","anio","semana"]`.
  - `fill_zeros(agg: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame` → grid + `casos`
    (0 donde no había registro), ordenado por `provincia, anio, semana`.

- [ ] **Step 1: Escribir el test que falla** (`tests/data/test_grid.py`)

```python
from satec.config import WEEKS
from satec.data.load import clean
from satec.data.aggregate import aggregate_weekly
from satec.data.grid import build_full_grid, fill_zeros


def test_grid_es_producto_cartesiano(sample_raw):
    agg = aggregate_weekly(clean(sample_raw))
    grid = build_full_grid(agg)
    n_prov = agg[["departamento", "provincia"]].drop_duplicates().shape[0]
    n_anios = agg["anio"].nunique()
    assert len(grid) == n_prov * n_anios * len(WEEKS)


def test_fill_zeros_imputa_y_conserva_conteos(sample_raw):
    agg = aggregate_weekly(clean(sample_raw))
    full = fill_zeros(agg, build_full_grid(agg))
    # HUARAZ 2010 s5 conserva 2 casos
    f = full[(full["provincia"] == "HUARAZ") & (full["anio"] == 2010)
             & (full["semana"] == 5)]
    assert int(f["casos"].iloc[0]) == 2
    # una semana sin registros queda en 0, sin nulos
    assert full["casos"].isna().sum() == 0
    assert (full["casos"] == 0).any()
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest tests/data/test_grid.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.data.grid'`

- [ ] **Step 3: Implementar `src/satec/data/grid.py`**

```python
"""Rejilla completa provincia x anio x semana con imputacion de ceros."""
import itertools
import pandas as pd

from satec.config import WEEKS


def build_full_grid(agg: pd.DataFrame) -> pd.DataFrame:
    provincias = agg[["departamento", "provincia"]].drop_duplicates()
    anios = sorted(agg["anio"].unique())
    combos = []
    for (_, dep, prov), anio, sem in itertools.product(
        provincias.itertuples(name=None), anios, WEEKS
    ):
        combos.append((dep, prov, anio, sem))
    return pd.DataFrame(
        combos, columns=["departamento", "provincia", "anio", "semana"])


def fill_zeros(agg: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    merged = grid.merge(
        agg, on=["departamento", "provincia", "anio", "semana"], how="left")
    merged["casos"] = merged["casos"].fillna(0).astype(int)
    return merged.sort_values(
        ["departamento", "provincia", "anio", "semana"]).reset_index(drop=True)
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest tests/data/test_grid.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/data/grid.py tests/data/test_grid.py
git commit -m "feat(data): rejilla completa e imputacion de ceros"
```

---

### Task 5: Selección de provincias endémicas

**Files:**
- Create: `src/satec/data/endemic.py`
- Test: `tests/data/test_endemic.py`

**Interfaces:**
- Consumes: salida de `aggregate_weekly()`.
- Produces: `select_endemic_provinces(agg, min_casos=10, min_anios=3) -> pd.DataFrame`
  → columnas `["departamento","provincia"]` de las provincias que acumulan
  `>= min_casos` casos en `>= min_anios` años distintos con al menos un caso.

- [ ] **Step 1: Escribir el test que falla** (`tests/data/test_endemic.py`)

```python
import pandas as pd
from satec.data.endemic import select_endemic_provinces


def _agg(rows):
    return pd.DataFrame(
        rows, columns=["departamento", "provincia", "anio", "semana", "casos"])


def test_endemic_filtra_por_umbrales():
    agg = _agg([
        ("ANCASH", "HUARAZ", 2010, 5, 8),
        ("ANCASH", "HUARAZ", 2011, 5, 8),
        ("ANCASH", "HUARAZ", 2012, 5, 8),   # 24 casos en 3 anios -> endemica
        ("LIMA", "HUARAL", 2010, 8, 1),     # 1 caso, 1 anio -> NO endemica
    ])
    out = select_endemic_provinces(agg, min_casos=10, min_anios=3)
    provincias = set(out["provincia"])
    assert "HUARAZ" in provincias
    assert "HUARAL" not in provincias
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest tests/data/test_endemic.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.data.endemic'`

- [ ] **Step 3: Implementar `src/satec/data/endemic.py`**

```python
"""Seleccion data-driven de provincias endemicas."""
import pandas as pd


def select_endemic_provinces(
    agg: pd.DataFrame, min_casos: int = 10, min_anios: int = 3
) -> pd.DataFrame:
    con_casos = agg[agg["casos"] > 0]
    resumen = con_casos.groupby(["departamento", "provincia"]).agg(
        total_casos=("casos", "sum"),
        anios_con_casos=("anio", "nunique"),
    ).reset_index()
    sel = resumen[(resumen["total_casos"] >= min_casos)
                  & (resumen["anios_con_casos"] >= min_anios)]
    return sel[["departamento", "provincia"]].reset_index(drop=True)
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest tests/data/test_endemic.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/data/endemic.py tests/data/test_endemic.py
git commit -m "feat(data): seleccion data-driven de provincias endemicas"
```

---

### Task 6: Canal endémico causal

**Files:**
- Create: `src/satec/data/endemic_channel.py`
- Test: `tests/data/test_endemic_channel.py`

**Interfaces:**
- Consumes: panel con ceros (salida de `fill_zeros()`), columnas
  `[departamento,provincia,anio,semana,casos]`.
- Produces: `compute_channel(panel, n_ref=5, min_ref=3) -> pd.DataFrame` → panel + columnas
  `["q1","q2","q3"]` (float, cuartiles de los casos de esa semana en los `n_ref`
  años previos de la misma provincia). Filas con menos de `min_ref` años de
  referencia → `q1=q2=q3=NaN`.

- [ ] **Step 1: Escribir el test que falla** (`tests/data/test_endemic_channel.py`)

```python
import numpy as np
import pandas as pd
from satec.data.endemic_channel import compute_channel


def _panel():
    # Una provincia, semana 5 fija, casos crecientes por anio
    rows = []
    for i, anio in enumerate(range(2010, 2017)):
        rows.append(("ANCASH", "HUARAZ", anio, 5, i))  # casos = 0..6
    return pd.DataFrame(
        rows, columns=["departamento", "provincia", "anio", "semana", "casos"])


def test_channel_nan_si_falta_historia():
    out = compute_channel(_panel(), n_ref=5, min_ref=3)
    # 2010 y 2011 no tienen >=3 anios previos
    early = out[out["anio"].isin([2010, 2011])]
    assert early[["q1", "q2", "q3"]].isna().all().all()


def test_channel_usa_solo_anios_previos():
    out = compute_channel(_panel(), n_ref=5, min_ref=3)
    # Para 2015 (casos previos en s5: 2010..2014 -> 0,1,2,3,4), Q2 = mediana = 2
    fila = out[(out["anio"] == 2015) & (out["semana"] == 5)]
    assert abs(float(fila["q2"].iloc[0]) - 2.0) < 1e-9
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest tests/data/test_endemic_channel.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.data.endemic_channel'`

- [ ] **Step 3: Implementar `src/satec/data/endemic_channel.py`**

```python
"""Canal endemico causal: cuartiles por semana usando solo anios previos."""
import numpy as np
import pandas as pd


def compute_channel(panel: pd.DataFrame, n_ref: int = 5,
                    min_ref: int = 3) -> pd.DataFrame:
    panel = panel.sort_values(
        ["departamento", "provincia", "semana", "anio"]).reset_index(drop=True)
    q1 = np.full(len(panel), np.nan)
    q2 = np.full(len(panel), np.nan)
    q3 = np.full(len(panel), np.nan)

    grupos = panel.groupby(["departamento", "provincia", "semana"], sort=False)
    for _, idx in grupos.groups.items():
        idx = list(idx)
        sub = panel.loc[idx].sort_values("anio")
        casos = sub["casos"].to_numpy()
        posiciones = sub.index.to_numpy()
        for j in range(len(sub)):
            ref = casos[max(0, j - n_ref):j]  # anios estrictamente previos
            if len(ref) >= min_ref:
                pos = posiciones[j]
                q1[pos], q2[pos], q3[pos] = np.percentile(ref, [25, 50, 75])

    panel = panel.copy()
    panel["q1"], panel["q2"], panel["q3"] = q1, q2, q3
    return panel
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest tests/data/test_endemic_channel.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/data/endemic_channel.py tests/data/test_endemic_channel.py
git commit -m "feat(data): canal endemico causal por cuartiles"
```

---

### Task 7: Etiquetado del target de brote a 4 semanas

**Files:**
- Create: `src/satec/data/target.py`
- Test: `tests/data/test_target.py`

**Interfaces:**
- Consumes: panel con canal (salida de `compute_channel()`).
- Produces: `label_outbreak(panel, horizon=4, min_cases=2) -> pd.DataFrame` → panel +
  columna `brote` (Int: 0/1, o `<NA>` cuando no hay ventana futura completa o falta
  canal en la ventana). Brote=1 si en alguna semana de `[t+1,t+4]`:
  `casos > q3` **y** `casos >= min_cases`.

- [ ] **Step 1: Escribir el test que falla** (`tests/data/test_target.py`)

```python
import pandas as pd
from satec.data.target import label_outbreak


def _panel_con_canal():
    # provincia unica, semanas 1..10 de un anio, con q3 constante = 1
    rows = []
    casos = [0, 0, 0, 0, 5, 0, 0, 0, 0, 0]  # pico en semana 5
    for i, c in enumerate(casos, start=1):
        rows.append(("ANCASH", "HUARAZ", 2015, i, c, 0.0, 0.0, 1.0))
    return pd.DataFrame(rows, columns=[
        "departamento", "provincia", "anio", "semana", "casos",
        "q1", "q2", "q3"])


def test_target_marca_brote_en_ventana_previa():
    out = label_outbreak(_panel_con_canal(), horizon=4, min_cases=2)
    # semana 1: ventana [2..5] incluye el pico (5 > q3=1 y >=2) -> brote=1
    assert int(out[out["semana"] == 1]["brote"].iloc[0]) == 1
    # semana 5: ventana [6..9] sin picos -> brote=0
    assert int(out[out["semana"] == 5]["brote"].iloc[0]) == 0


def test_target_nulo_si_ventana_incompleta():
    out = label_outbreak(_panel_con_canal(), horizon=4, min_cases=2)
    # semanas 7..10 no tienen 4 semanas futuras dentro del panel
    assert out[out["semana"] == 10]["brote"].isna().all()
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest tests/data/test_target.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.data.target'`

- [ ] **Step 3: Implementar `src/satec/data/target.py`**

```python
"""Etiquetado del target de brote en la ventana [t+1, t+horizon]."""
import numpy as np
import pandas as pd


def label_outbreak(panel: pd.DataFrame, horizon: int = 4,
                   min_cases: int = 2) -> pd.DataFrame:
    panel = panel.sort_values(
        ["departamento", "provincia", "anio", "semana"]).reset_index(drop=True)
    brote = np.full(len(panel), np.nan)

    es_epi = ((panel["casos"] > panel["q3"]) &
              (panel["casos"] >= min_cases)).to_numpy()
    q3_na = panel["q3"].isna().to_numpy()

    grupos = panel.groupby(["departamento", "provincia"], sort=False)
    for _, idx in grupos.groups.items():
        idx = list(idx)
        for k in range(len(idx)):
            fin = k + horizon
            if fin >= len(idx):
                continue  # ventana futura incompleta -> NaN
            ventana = idx[k + 1:fin + 1]
            if any(q3_na[p] for p in ventana):
                continue  # sin canal en la ventana -> NaN
            brote[idx[k]] = 1.0 if any(es_epi[p] for p in ventana) else 0.0

    panel = panel.copy()
    panel["brote"] = pd.array(brote, dtype="Int64")
    return panel
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest tests/data/test_target.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/data/target.py tests/data/test_target.py
git commit -m "feat(data): etiquetado del target de brote a 4 semanas"
```

---

### Task 8: Características temporales (lags, medias móviles, estacionalidad)

**Files:**
- Create: `src/satec/features/build.py`
- Test: `tests/features/test_build.py`
- Create: `tests/features/__init__.py`

**Interfaces:**
- Consumes: panel ordenado con `casos` (salida de `fill_zeros()` o posterior).
- Produces: `add_features(panel, lags=(1,2,4), windows=(4,8)) -> pd.DataFrame` → panel +
  `casos_lag1, casos_lag2, casos_lag4, roll_mean4, roll_mean8, sin_semana,
  cos_semana`. Los lags/rolling se calculan **por provincia** en orden temporal;
  los valores iniciales sin historia quedan en 0.

- [ ] **Step 1: Escribir el test que falla** (`tests/features/test_build.py`)

```python
import numpy as np
import pandas as pd
from satec.features.build import add_features


def _panel():
    rows = []
    for i, sem in enumerate(range(1, 11)):
        rows.append(("ANCASH", "HUARAZ", 2015, sem, i))  # casos = 0..9
    return pd.DataFrame(
        rows, columns=["departamento", "provincia", "anio", "semana", "casos"])


def test_lag_features():
    out = add_features(_panel(), lags=(1, 2, 4), windows=(4, 8))
    fila = out[out["semana"] == 5].iloc[0]
    assert fila["casos"] == 4
    assert fila["casos_lag1"] == 3
    assert fila["casos_lag4"] == 0  # casos de la semana 1


def test_seasonal_features_en_rango():
    out = add_features(_panel())
    assert out["sin_semana"].between(-1, 1).all()
    assert out["cos_semana"].between(-1, 1).all()
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest tests/features/test_build.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.features.build'`

- [ ] **Step 3: Implementar `src/satec/features/build.py`**

```python
"""Caracteristicas temporales por provincia."""
import numpy as np
import pandas as pd


def add_features(panel: pd.DataFrame, lags=(1, 2, 4),
                 windows=(4, 8)) -> pd.DataFrame:
    panel = panel.sort_values(
        ["departamento", "provincia", "anio", "semana"]).reset_index(drop=True)
    grp = panel.groupby(["departamento", "provincia"], sort=False)["casos"]

    for L in lags:
        panel[f"casos_lag{L}"] = grp.shift(L).fillna(0)
    for W in windows:
        # rolling por grupo (transform realinea al panel; evita cruzar provincias)
        panel[f"roll_mean{W}"] = grp.transform(
            lambda s: s.shift(1).rolling(W, min_periods=1).mean()).fillna(0)

    ang = 2 * np.pi * panel["semana"] / 52.0
    panel["sin_semana"] = np.sin(ang)
    panel["cos_semana"] = np.cos(ang)
    return panel
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest tests/features/test_build.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/features/build.py tests/features/__init__.py tests/features/test_build.py
git commit -m "feat(features): lags, medias moviles y estacionalidad"
```

---

### Task 9: Orquestador del pipeline y dataset reproducible

**Files:**
- Create: `src/satec/data/build_dataset.py`
- Test: `tests/data/test_build_dataset.py`
- Modify: copiar el CSV del MINSA a `data/raw/carrion_minsa_2000_2024.csv`

**Interfaces:**
- Consumes: todos los módulos anteriores + `DEFAULT_PARAMS`.
- Produces:
  - `build_dataset(raw_path, params=DEFAULT_PARAMS) -> pd.DataFrame` (en memoria).
  - `main(raw_path, out_path)` → escribe `data/processed/dataset.parquet`.
  - Dataset final: filas de provincias endémicas con `brote` no nulo, columnas de
    features + `casos,q1,q2,q3,brote` + claves `departamento,provincia,anio,semana`.

- [ ] **Step 1: Copiar el CSV crudo del MINSA al proyecto**

Run:
```bash
cp "/c/Users/Usuario/Desktop/IA/Sesion8/datos_abiertos_vigilancia_enfermedad_carrion_2000_2024 (1).csv" \
   "/c/Users/Usuario/Desktop/IA/SATEC-Carrion/data/raw/carrion_minsa_2000_2024.csv"
```
Expected: el archivo existe en `data/raw/`.

- [ ] **Step 2: Escribir el test que falla** (`tests/data/test_build_dataset.py`)

```python
from satec.data.build_dataset import build_dataset


def test_build_dataset_desde_fixture(tmp_path, sample_raw):
    raw = tmp_path / "raw.csv"
    sample_raw.to_csv(raw, index=False)
    params = {"horizon": 1, "n_ref": 1, "min_ref": 1, "min_cases": 1,
              "endemic_min_casos": 1, "endemic_min_anios": 1}
    df = build_dataset(str(raw), params=params)
    assert len(df) > 0
    # columnas clave presentes y sin brote nulo
    for col in ["departamento", "provincia", "anio", "semana", "casos",
                "brote", "casos_lag1", "sin_semana"]:
        assert col in df.columns
    assert df["brote"].isna().sum() == 0
```

- [ ] **Step 3: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest tests/data/test_build_dataset.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.data.build_dataset'`

- [ ] **Step 4: Implementar `src/satec/data/build_dataset.py`**

```python
"""Orquestador del pipeline de datos núcleo de SATEC."""
import pandas as pd

from satec.config import DEFAULT_PARAMS
from satec.data.load import load_raw, clean
from satec.data.aggregate import aggregate_weekly
from satec.data.grid import build_full_grid, fill_zeros
from satec.data.endemic import select_endemic_provinces
from satec.data.endemic_channel import compute_channel
from satec.data.target import label_outbreak
from satec.features.build import add_features


def build_dataset(raw_path: str, params: dict = DEFAULT_PARAMS) -> pd.DataFrame:
    df = clean(load_raw(raw_path))
    agg = aggregate_weekly(df)

    endemic = select_endemic_provinces(
        agg, params["endemic_min_casos"], params["endemic_min_anios"])
    agg = agg.merge(endemic, on=["departamento", "provincia"], how="inner")

    panel = fill_zeros(agg, build_full_grid(agg))
    panel = compute_channel(panel, params["n_ref"], params["min_ref"])
    panel = label_outbreak(panel, params["horizon"], params["min_cases"])
    panel = add_features(panel)

    return panel.dropna(subset=["brote"]).reset_index(drop=True)


def main(raw_path: str, out_path: str) -> None:
    df = build_dataset(raw_path)
    df.to_parquet(out_path, index=False)
    print(f"[OK] dataset: {df.shape} -> {out_path}")
    print(f"  provincias endemicas: {df['provincia'].nunique()}")
    print(f"  tasa de brotes: {df['brote'].mean():.3f}")


if __name__ == "__main__":
    main("data/raw/carrion_minsa_2000_2024.csv",
         "data/processed/dataset.parquet")
```

- [ ] **Step 5: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest tests/data/test_build_dataset.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Ejecutar el pipeline sobre los datos reales (smoke test)**

Run:
```bash
mkdir -p "/c/Users/Usuario/Desktop/IA/SATEC-Carrion/data/processed"
cd "/c/Users/Usuario/Desktop/IA/SATEC-Carrion" && PYTHONPATH=src py -3.12 -m satec.data.build_dataset
```
Expected: imprime la forma del dataset, nº de provincias endémicas y tasa de brotes (un valor entre 0 y 1, p. ej. 0.05–0.30). Si la tasa es 0 o 1, revisar parámetros del canal/target.

- [ ] **Step 7: Ejecutar la suite completa**

Run: `cd "/c/Users/Usuario/Desktop/IA/SATEC-Carrion" && py -3.12 -m pytest`
Expected: PASS (todos los tests verdes).

- [ ] **Step 8: Commit**

```bash
git add src/satec/data/build_dataset.py tests/data/test_build_dataset.py
git commit -m "feat(data): orquestador del pipeline y dataset reproducible"
```

---

## Notas para el ejecutor

- Ejecutar siempre con `py -3.12`. Para que `import satec...` funcione, correr pytest y los módulos **desde la raíz del proyecto** (`SATEC-Carrion/`). Si hubiera problemas de importación, añadir un `conftest.py` en la raíz o instalar el paquete en modo editable; preferimos correr desde la raíz para no añadir empaquetado en esta fase.
- El dataset `data/processed/dataset.parquet` está en `.gitignore` (derivado, regenerable). No se commitea.
- El CSV `data/raw/carrion_minsa_2000_2024.csv` SÍ se versiona (fuente reproducible).

## Próximo plan

**Plan 2 — Enriquecimiento (clima NASA POWER + población INEI).** Antes de escribirlo, verificar disponibilidad y formato de las fuentes (tarea de exploración previa). El dataset de este plan es la base sobre la que se unirán esas variables.
