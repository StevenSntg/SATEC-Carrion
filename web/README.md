# SATEC — Aplicación web

Mapa de riesgo de brotes de la enfermedad de Carrión por provincia del Perú y
comparador **Red Neuronal vs Árbol de Decisión**. Página estática (HTML + CSS +
JavaScript con Leaflet); los modelos se exportan desde Python (la web no reentrena).

## Regenerar datos y modelos

Desde la raíz del proyecto (`SATEC-Carrion/`), con Python 3.12 y el dataset
enriquecido ya construido (`data/processed/dataset_enriched.parquet`):

```bash
# Exportar Red Neuronal (TF.js) y Árbol de Decisión (JSON):
PYTHONPATH=src TF_USE_LEGACY_KERAS=1 py -3.12 -c "from satec.web_export.export_models import export_all; export_all('.')"

# Precomputar el riesgo por provincia (última semana) + GeoJSON endémico:
PYTHONPATH=src TF_USE_LEGACY_KERAS=1 py -3.12 -c "from satec.web_export.risk_map import build_risk_json; build_risk_json('.')"
```

Esto genera `web/public/models/` (modelos) y `web/public/data/` (riesgo + GeoJSON).

## Probar localmente

```bash
py -3.12 -m http.server --directory web/public 8000
# abrir http://localhost:8000
```

## Desplegar (Vercel)

`vercel.json` publica únicamente `public/`. Desde la carpeta `web/`: `vercel --prod`.

## Estructura

```
web/
├── vercel.json
└── public/
    ├── index.html · styles.css · app.js     # la página
    ├── data/riesgo.json                      # riesgo por provincia (última semana)
    ├── data/provincias.geojson               # límites de provincias endémicas
    └── models/                               # Red Neuronal (TF.js) y Árbol (JSON)
```

> Herramienta de apoyo a la vigilancia. No sustituye el diagnóstico clínico ni la
> confirmación de laboratorio.
