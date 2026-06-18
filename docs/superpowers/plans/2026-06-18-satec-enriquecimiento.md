# Plan 2 — Enriquecimiento (clima + población) — SATEC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enriquecer el dataset del Plan 1 (provincia×semana) con variables **climáticas** (NASA POWER) y **poblacionales** (INEI), homologando por **ubigeo de provincia**, para añadir predictores y tasas por 100.000 habitantes.

**Architecture:** Módulos puros bajo `src/satec/geo` y `src/satec/enrich`, testeados de forma aislada con fixtures (sin red). Un módulo de descarga con caché en disco aísla las llamadas de red (NASA POWER) del resto del pipeline. Un orquestador une todo al dataset del Plan 1.

**Tech Stack:** Python 3.12 (`py -3.12`), pandas 3.0.3, numpy 1.26.4, pytest 9.0.3. Red: `urllib`/`curl` (NASA POWER, GitHub raw). Sin shapely (centroide calculado a mano).

## Global Constraints

- Intérprete: **Python 3.12** (`py -3.12`). Correr pytest desde la raíz (pythonpath=src ya configurado).
- Clave de homologación entre fuentes: **ubigeo de provincia** = primeros 4 dígitos del ubigeo distrital del MINSA. El GeoJSON usa `FIRST_IDPR`.
- Variables climáticas NASA POWER: `PRECTOTCORR` (precipitación mm/día), `T2M` (temperatura °C), `RH2M` (humedad %).
- Agregación diaria → **semana epidemiológica**: precipitación = **suma**; temperatura y humedad = **media**. La semana se aproxima con ISO week (`isocalendar().week`, clip 53→52) — diferencia menor con la definición CDC, declarada como limitación.
- Rezagos climáticos: features en la semana `t` usan clima de `t`, `t-4`, `t-8` y medias móviles (4, 8). El vector *Lutzomyia* responde al clima con retraso.
- Datos descargados se cachean en `data/raw/` (clima, geojson) y `data/interim/`. Los tests NO usan red (fixtures).
- `fill_value` de NASA POWER = `-999.0` → tratar como NaN.
- Población: intentar INEI; si el formato es intratable, **plan B**: población provincial constante del Censo 2017 (documentada como limitación). Las tasas son normalización, no el predictor central.
- Commits: cuenta **stevensntg**, **sin** coautoría de Claude.

---

### Task 1: Carga del GeoJSON de provincias y centroides

**Files:**
- Create: `src/satec/geo/__init__.py`
- Create: `src/satec/geo/provinces.py`
- Test: `tests/geo/__init__.py`, `tests/geo/test_provinces.py`

**Interfaces:**
- Produces:
  - `load_provinces_geojson(path) -> list[dict]` (features).
  - `province_centroid(feature) -> tuple[float, float]` → `(lon, lat)` promedio de los vértices del anillo exterior (soporta Polygon y MultiPolygon).
  - `provinces_table(geojson) -> pd.DataFrame` → columnas `["ubigeo_prov","nombre_prov","lon","lat"]` (`ubigeo_prov` = str de 4 dígitos desde `FIRST_IDPR`).

- [ ] **Step 1: Escribir el test que falla** (`tests/geo/test_provinces.py`)

```python
from satec.geo.provinces import province_centroid, provinces_table


def _feature():
    return {
        "type": "Feature",
        "properties": {"NOMBPROV": "HUARAZ", "FIRST_IDPR": "0201"},
        "geometry": {"type": "Polygon",
                     "coordinates": [[[-77.0, -9.0], [-78.0, -9.0],
                                      [-78.0, -10.0], [-77.0, -10.0],
                                      [-77.0, -9.0]]]},
    }


def test_centroid_promedia_vertices():
    lon, lat = province_centroid(_feature())
    assert abs(lon - (-77.5)) < 1e-6
    assert abs(lat - (-9.5)) < 1e-6


def test_provinces_table_columnas():
    df = provinces_table({"features": [_feature()]})
    assert list(df.columns) == ["ubigeo_prov", "nombre_prov", "lon", "lat"]
    assert df["ubigeo_prov"].iloc[0] == "0201"
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/geo/test_provinces.py" -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.geo'`

- [ ] **Step 3: Implementar `src/satec/geo/provinces.py`** (y `__init__.py` vacíos)

```python
"""Carga de límites provinciales (GeoJSON) y cálculo de centroides."""
import json
import pandas as pd


def load_provinces_geojson(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _exterior_rings(geometry: dict):
    t = geometry["type"]
    coords = geometry["coordinates"]
    if t == "Polygon":
        return [coords[0]]
    if t == "MultiPolygon":
        return [poly[0] for poly in coords]
    raise ValueError(f"Geometría no soportada: {t}")


def province_centroid(feature: dict) -> tuple:
    pts = []
    for ring in _exterior_rings(feature["geometry"]):
        pts.extend(ring[:-1])  # excluye el vértice de cierre (igual al primero)
    lon = sum(p[0] for p in pts) / len(pts)
    lat = sum(p[1] for p in pts) / len(pts)
    return (lon, lat)


def provinces_table(geojson: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson["features"]:
        props = feat["properties"]
        lon, lat = province_centroid(feat)
        rows.append({
            "ubigeo_prov": str(props["FIRST_IDPR"]).zfill(4),
            "nombre_prov": str(props["NOMBPROV"]).strip().upper(),
            "lon": lon, "lat": lat,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/geo/test_provinces.py" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Descargar el GeoJSON real (red)**

Run:
```bash
curl -sL "https://raw.githubusercontent.com/juaneladio/peru-geojson/master/peru_provincial_simple.geojson" \
  -o "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/data/raw/peru_provincial.geojson"
py -3.12 -c "import json; d=json.load(open(r'C:/Users/Usuario/Desktop/IA/SATEC-Carrion/data/raw/peru_provincial.geojson',encoding='utf-8')); f=d['features'][0]['properties']; print('n_features', len(d['features'])); print('props', list(f.keys()))"
```
Expected: ~196 features; las propiedades incluyen `NOMBPROV` y `FIRST_IDPR`. **Si los nombres difieren, ajustar `provinces_table` y este plan.**

- [ ] **Step 6: Commit**

```bash
git add src/satec/geo tests/geo data/raw/peru_provincial.geojson
git commit -m "feat(geo): carga de GeoJSON provincial y centroides"
```

---

### Task 2: Homologación nombre de provincia → ubigeo de provincia

**Files:**
- Create: `src/satec/data/ubigeo.py`
- Test: `tests/data/test_ubigeo.py`

**Interfaces:**
- Consumes: CSV crudo del MINSA (tiene `ubigeo` distrital de 6 dígitos + `departamento`,`provincia`).
- Produces: `province_ubigeo_map(raw_df) -> pd.DataFrame` → `["departamento","provincia","ubigeo_prov"]`,
  con `ubigeo_prov` = los 4 primeros dígitos del ubigeo distrital **modal** por (departamento, provincia).

- [ ] **Step 1: Escribir el test que falla** (`tests/data/test_ubigeo.py`)

```python
import pandas as pd
from satec.data.ubigeo import province_ubigeo_map


def test_ubigeo_prov_modal():
    raw = pd.DataFrame({
        "departamento": ["ANCASH", "ANCASH", "ANCASH"],
        "provincia": ["HUARAZ", "HUARAZ", "HUARAZ"],
        "ubigeo": ["020101", "020102", "020101"],  # prov 0201 modal
    })
    out = province_ubigeo_map(raw)
    fila = out[(out["provincia"] == "HUARAZ")]
    assert fila["ubigeo_prov"].iloc[0] == "0201"
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/data/test_ubigeo.py" -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.data.ubigeo'`

- [ ] **Step 3: Implementar `src/satec/data/ubigeo.py`**

```python
"""Homologación de (departamento, provincia) -> ubigeo de provincia."""
import pandas as pd


def province_ubigeo_map(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["departamento"] = df["departamento"].astype(str).str.strip().str.upper()
    df["provincia"] = df["provincia"].astype(str).str.strip().str.upper()
    df["ubigeo_prov"] = df["ubigeo"].astype(str).str.zfill(6).str[:4]

    def _modal(s):
        return s.mode().iloc[0]

    out = (df.groupby(["departamento", "provincia"])["ubigeo_prov"]
             .agg(_modal).reset_index())
    return out
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/data/test_ubigeo.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/data/ubigeo.py tests/data/test_ubigeo.py
git commit -m "feat(data): homologacion nombre de provincia a ubigeo"
```

---

### Task 3: Parser de NASA POWER (diario → semanal)

**Files:**
- Create: `src/satec/enrich/__init__.py`
- Create: `src/satec/enrich/climate.py`
- Test: `tests/enrich/__init__.py`, `tests/enrich/test_climate.py`

**Interfaces:**
- Consumes: respuesta JSON de NASA POWER (dict).
- Produces:
  - `parse_power_json(payload) -> pd.DataFrame` → `["fecha","prec","temp","hum"]` (`fecha` datetime; `-999.0`→NaN).
  - `to_weekly(daily, anio_col=True) -> pd.DataFrame` → `["anio","semana","prec","temp","hum"]`
    (prec=suma semanal; temp,hum=media; semana = ISO week con 53→52).

- [ ] **Step 1: Escribir el test que falla** (`tests/enrich/test_climate.py`)

```python
from satec.enrich.climate import parse_power_json, to_weekly


def _payload():
    return {"properties": {"parameter": {
        "PRECTOTCORR": {"20100104": 1.0, "20100105": 2.0, "20100106": -999.0},
        "T2M": {"20100104": 10.0, "20100105": 12.0, "20100106": 11.0},
        "RH2M": {"20100104": 60.0, "20100105": 62.0, "20100106": 64.0},
    }}}


def test_parse_power_json_marca_fill_value_como_nan():
    df = parse_power_json(_payload())
    assert list(df.columns) == ["fecha", "prec", "temp", "hum"]
    assert df["prec"].isna().sum() == 1  # -999.0 -> NaN


def test_to_weekly_suma_prec_y_promedia_temp():
    weekly = to_weekly(parse_power_json(_payload()))
    # los 3 dias caen en la misma semana ISO de 2010
    fila = weekly.iloc[0]
    assert abs(fila["prec"] - 3.0) < 1e-9          # 1+2 (NaN ignorado)
    assert abs(fila["temp"] - 11.0) < 1e-9         # media de 10,12,11
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_climate.py" -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'satec.enrich'`

- [ ] **Step 3: Implementar `src/satec/enrich/climate.py`** (y `__init__.py`)

```python
"""Parseo y agregación semanal de datos climáticos de NASA POWER."""
import numpy as np
import pandas as pd

_PARAMS = {"PRECTOTCORR": "prec", "T2M": "temp", "RH2M": "hum"}
_FILL = -999.0


def parse_power_json(payload: dict) -> pd.DataFrame:
    param = payload["properties"]["parameter"]
    fechas = sorted(param["PRECTOTCORR"].keys())
    data = {"fecha": pd.to_datetime(fechas, format="%Y%m%d")}
    for src, dst in _PARAMS.items():
        serie = [param[src][f] for f in fechas]
        data[dst] = [np.nan if v == _FILL else v for v in serie]
    return pd.DataFrame(data)


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    iso = df["fecha"].dt.isocalendar()
    df["anio"] = iso["year"].astype(int)
    df["semana"] = iso["week"].astype(int).clip(upper=52)
    agg = df.groupby(["anio", "semana"], as_index=False).agg(
        prec=("prec", "sum"), temp=("temp", "mean"), hum=("hum", "mean"))
    return agg
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_climate.py" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/enrich/__init__.py src/satec/enrich/climate.py tests/enrich
git commit -m "feat(enrich): parser de NASA POWER y agregacion semanal"
```

---

### Task 4: Descarga de clima con caché (red)

**Files:**
- Create: `src/satec/enrich/climate_download.py`
- Test: `tests/enrich/test_climate_download.py` (solo prueba el caché, sin red)

**Interfaces:**
- Produces:
  - `power_url(lon, lat, start, end) -> str`.
  - `fetch_point(lon, lat, start="20000101", end="20241231", cache_dir=...) -> dict`
    (si existe `<cache_dir>/<lon>_<lat>.json`, lo lee; si no, descarga con `urllib` y lo guarda).
  - `download_all(provinces_df, cache_dir) -> None` (itera filas, llama `fetch_point`).

- [ ] **Step 1: Escribir el test que falla** (`tests/enrich/test_climate_download.py`)

```python
import json
from satec.enrich.climate_download import power_url, fetch_point


def test_power_url_incluye_parametros():
    url = power_url(-77.5, -9.5, "20000101", "20001231")
    assert "PRECTOTCORR" in url and "T2M" in url and "RH2M" in url
    assert "longitude=-77.5" in url and "latitude=-9.5" in url


def test_fetch_point_usa_cache(tmp_path):
    # pre-creamos el archivo de cache -> no debe tocar la red
    cached = {"properties": {"parameter": {"PRECTOTCORR": {}}}}
    (tmp_path / "-77.5_-9.5.json").write_text(json.dumps(cached))
    out = fetch_point(-77.5, -9.5, cache_dir=str(tmp_path))
    assert out == cached
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_climate_download.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/enrich/climate_download.py`**

```python
"""Descarga de clima de NASA POWER por punto, con caché en disco."""
import json
import os
import time
import urllib.request

_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
_PARAMS = "PRECTOTCORR,T2M,RH2M"


def power_url(lon: float, lat: float, start: str, end: str) -> str:
    return (f"{_BASE}?parameters={_PARAMS}&community=AG"
            f"&longitude={lon}&latitude={lat}"
            f"&start={start}&end={end}&format=JSON")


def fetch_point(lon, lat, start="20000101", end="20241231",
                cache_dir="data/raw/climate") -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{lon}_{lat}.json")
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    url = power_url(lon, lat, start, end)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


def download_all(provinces_df, cache_dir="data/raw/climate") -> None:
    n = len(provinces_df)
    for i, row in enumerate(provinces_df.itertuples(), 1):
        fetch_point(row.lon, row.lat, cache_dir=cache_dir)
        print(f"  [{i}/{n}] ubigeo {row.ubigeo_prov} listo")
        time.sleep(0.5)  # cortesía con la API
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_climate_download.py" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/enrich/climate_download.py tests/enrich/test_climate_download.py
git commit -m "feat(enrich): descarga de clima NASA POWER con cache"
```

---

### Task 5: Features climáticas (rezagos y medias móviles)

**Files:**
- Create: `src/satec/enrich/climate_features.py`
- Test: `tests/enrich/test_climate_features.py`

**Interfaces:**
- Consumes: panel del dataset (provincia×anio×semana) ya unido con `prec,temp,hum`.
- Produces: `add_climate_features(panel, lags=(4, 8), windows=(4, 8)) -> pd.DataFrame` →
  añade `prec_lag4, prec_lag8, temp_lag4, temp_lag8, hum_lag4, hum_lag8,
  prec_roll4, prec_roll8, temp_roll4, temp_roll8, hum_roll4, hum_roll8`,
  calculadas **por ubigeo_prov** en orden temporal (NaN inicial → 0).

- [ ] **Step 1: Escribir el test que falla** (`tests/enrich/test_climate_features.py`)

```python
import pandas as pd
from satec.enrich.climate_features import add_climate_features


def _panel():
    rows = []
    for i, sem in enumerate(range(1, 13)):
        rows.append(("0201", 2015, sem, float(i), 10.0 + i, 60.0))
    return pd.DataFrame(rows, columns=[
        "ubigeo_prov", "anio", "semana", "prec", "temp", "hum"])


def test_lag4_de_precipitacion():
    out = add_climate_features(_panel(), lags=(4, 8), windows=(4, 8))
    fila = out[out["semana"] == 9].iloc[0]
    # prec en semana 9 (i=8) = 8.0; prec_lag4 = prec semana 5 (i=4) = 4.0
    assert abs(fila["prec_lag4"] - 4.0) < 1e-9
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_climate_features.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/enrich/climate_features.py`**

```python
"""Rezagos y medias móviles de variables climáticas, por provincia."""
import pandas as pd

_VARS = ["prec", "temp", "hum"]


def add_climate_features(panel: pd.DataFrame, lags=(4, 8),
                         windows=(4, 8)) -> pd.DataFrame:
    panel = panel.sort_values(
        ["ubigeo_prov", "anio", "semana"]).reset_index(drop=True)
    for var in _VARS:
        grp = panel.groupby("ubigeo_prov", sort=False)[var]
        for L in lags:
            panel[f"{var}_lag{L}"] = grp.shift(L).fillna(0)
        for W in windows:
            panel[f"{var}_roll{W}"] = grp.transform(
                lambda s: s.shift(1).rolling(W, min_periods=1).mean()).fillna(0)
    return panel
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_climate_features.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/enrich/climate_features.py tests/enrich/test_climate_features.py
git commit -m "feat(enrich): rezagos y medias moviles climaticas"
```

---

### Task 6: Población y tasas por 100.000

**Files:**
- Create: `src/satec/enrich/population.py`
- Test: `tests/enrich/test_population.py`

**Interfaces:**
- Consumes: panel con `ubigeo_prov, anio, casos` + tabla de población
  `["ubigeo_prov","anio","poblacion"]`.
- Produces:
  - `attach_rate(panel, pob_df) -> pd.DataFrame` → añade `poblacion` y
    `tasa = casos / poblacion * 100000` (si falta población → `poblacion` NaN, `tasa` NaN).

- [ ] **Step 1: Escribir el test que falla** (`tests/enrich/test_population.py`)

```python
import pandas as pd
from satec.enrich.population import attach_rate


def test_attach_rate_calcula_por_100k():
    panel = pd.DataFrame({"ubigeo_prov": ["0201"], "anio": [2015], "casos": [5]})
    pob = pd.DataFrame({"ubigeo_prov": ["0201"], "anio": [2015],
                        "poblacion": [100000]})
    out = attach_rate(panel, pob)
    assert abs(out["tasa"].iloc[0] - 5.0) < 1e-9  # 5/100000*100000 = 5
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_population.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/enrich/population.py`**

```python
"""Unión de población provincial y cálculo de tasas por 100.000 hab."""
import pandas as pd


def attach_rate(panel: pd.DataFrame, pob_df: pd.DataFrame) -> pd.DataFrame:
    out = panel.merge(pob_df, on=["ubigeo_prov", "anio"], how="left")
    out["tasa"] = out["casos"] / out["poblacion"] * 100000
    return out
```

- [ ] **Step 4: Ejecutar para ver que pasa**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_population.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/satec/enrich/population.py tests/enrich/test_population.py
git commit -m "feat(enrich): poblacion provincial y tasas por 100k"
```

---

### Task 7: Orquestador del enriquecimiento

**Files:**
- Create: `src/satec/enrich/build_enriched.py`
- Test: `tests/enrich/test_build_enriched.py`

**Interfaces:**
- Consumes: dataset del Plan 1 + tabla de provincias (geo) + caché de clima + población.
- Produces:
  - `attach_ubigeo(dataset, ubigeo_map) -> pd.DataFrame` (añade `ubigeo_prov` por nombre).
  - `weekly_climate_for_provinces(prov_df, cache_dir) -> pd.DataFrame`
    → `["ubigeo_prov","anio","semana","prec","temp","hum"]` (parsea caché por provincia).
  - `build_enriched(dataset, ubigeo_map, prov_df, cache_dir, pob_df=None) -> pd.DataFrame`
    (añade ubigeo_prov vía `ubigeo_map`, une clima semanal + features climáticas, y tasas si hay población).
  - `main()` → lee `data/processed/dataset.parquet` y escribe `data/processed/dataset_enriched.parquet`.

- [ ] **Step 1: Escribir el test que falla** (`tests/enrich/test_build_enriched.py`)

```python
import pandas as pd
from satec.enrich.build_enriched import attach_ubigeo


def test_attach_ubigeo_por_nombre():
    dataset = pd.DataFrame({"departamento": ["ANCASH"], "provincia": ["HUARAZ"],
                            "anio": [2015], "semana": [5], "casos": [3]})
    umap = pd.DataFrame({"departamento": ["ANCASH"], "provincia": ["HUARAZ"],
                         "ubigeo_prov": ["0201"]})
    out = attach_ubigeo(dataset, umap)
    assert out["ubigeo_prov"].iloc[0] == "0201"
```

- [ ] **Step 2: Ejecutar para ver el fallo**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_build_enriched.py" -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `src/satec/enrich/build_enriched.py`**

```python
"""Orquestador del enriquecimiento: clima + población sobre el dataset base."""
import json
import os
import pandas as pd

from satec.enrich.climate import parse_power_json, to_weekly
from satec.enrich.climate_features import add_climate_features
from satec.enrich.population import attach_rate


def attach_ubigeo(dataset: pd.DataFrame, ubigeo_map: pd.DataFrame) -> pd.DataFrame:
    """Añade ubigeo_prov al dataset uniendo por (departamento, provincia)."""
    return dataset.merge(ubigeo_map, on=["departamento", "provincia"], how="left")


def weekly_climate_for_provinces(prov_df: pd.DataFrame,
                                 cache_dir: str) -> pd.DataFrame:
    """Parsea el caché de clima por provincia -> panel semanal con ubigeo_prov."""
    partes = []
    for row in prov_df.itertuples():
        cache_file = os.path.join(cache_dir, f"{row.lon}_{row.lat}.json")
        if not os.path.exists(cache_file):
            continue
        with open(cache_file, encoding="utf-8") as f:
            payload = json.load(f)
        weekly = to_weekly(parse_power_json(payload))
        weekly["ubigeo_prov"] = row.ubigeo_prov
        partes.append(weekly)
    return pd.concat(partes, ignore_index=True)


def build_enriched(dataset, ubigeo_map, prov_df, cache_dir,
                   pob_df=None) -> pd.DataFrame:
    df = attach_ubigeo(dataset, ubigeo_map)            # añade ubigeo_prov
    clima = weekly_climate_for_provinces(prov_df, cache_dir)
    df = df.merge(clima, on=["ubigeo_prov", "anio", "semana"], how="left")
    df = add_climate_features(df)
    if pob_df is not None:
        df = attach_rate(df, pob_df)
    return df
```

> Nota para el ejecutor: `build_enriched` une el dataset con el clima por
> `ubigeo_prov` (obtenido del `ubigeo_map` de la Task 2), NO por nombre — así se
> evitan discrepancias de tildes/grafías. El `prov_df` (Task 1, `provinces_table`)
> aporta los centroides y el `ubigeo_prov` del lado del clima.

- [ ] **Step 4: Ejecutar el test (solo `attach_ubigeo`)**

Run: `py -3.12 -m pytest "C:/Users/Usuario/Desktop/IA/SATEC-Carrion/tests/enrich/test_build_enriched.py" -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Descargar clima real y smoke test de extremo a extremo**

```bash
REPO="C:/Users/Usuario/Desktop/IA/SATEC-Carrion"
# 1) tabla de provincias endemicas con centroides + ubigeo (script ad-hoc)
# 2) download_all sobre esas provincias (cachea ~61 JSON)
# 3) build_enriched sobre data/processed/dataset.parquet
# Implementar main() concreto que: carga dataset, ubigeo_map (del raw),
# provinces_table (geojson), filtra a provincias del dataset, descarga clima,
# construye enriquecido y guarda dataset_enriched.parquet.
```
Expected: `data/processed/dataset_enriched.parquet` con columnas de clima no nulas en la mayoría de filas. Reportar % de filas con clima.

- [ ] **Step 6: Commit**

```bash
git add src/satec/enrich/build_enriched.py tests/enrich/test_build_enriched.py
git commit -m "feat(enrich): orquestador de enriquecimiento climatico"
```

---

## Notas para el ejecutor

- **Población (Task 6):** la unión por `(ubigeo_prov, anio)` es el contrato. Conseguir la tabla `["ubigeo_prov","anio","poblacion"]` desde INEI puede requerir parsear un Excel; si resulta intratable en esta iteración, generar la tabla con población del Censo 2017 replicada por año (constante) y declararlo como limitación en el paper. El modelado (Plan 3) puede usar `casos` y/o `tasa`.
- **`main()` de Task 7:** debe quedar concreto al ejecutar (carga dataset base, construye `ubigeo_map` desde el raw, `provinces_table` desde el geojson, filtra a las provincias presentes en el dataset, descarga clima, enriquece y guarda). Si la unión por nombre da discrepancias, homologar por `ubigeo_prov`.
- El parquet enriquecido está en `.gitignore` (derivado). Los JSON de clima en `data/raw/climate/` SÍ se versionan (caché reproducible) salvo que pesen demasiado; si superan ~50 MB, añadir a `.gitignore` y documentar el script de descarga.

## Próximo plan

**Plan 3 — Modelado y evaluación** (AD/RN/ensembles + baselines, validación temporal, métricas, calibración, SHAP) sobre `dataset_enriched.parquet`.
