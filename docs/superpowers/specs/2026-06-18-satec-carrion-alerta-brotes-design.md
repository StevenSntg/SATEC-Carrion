# Diseño — SATEC: Sistema de Alerta Temprana de la Enfermedad de Carrión

| Campo | Valor |
|---|---|
| **Fecha** | 2026-06-18 |
| **Autores** | Jaqueline Alvarez Rocca · Carlos Meza Pelaez · Carlos Steven Santiago Flores |
| **Afiliación** | E.P. de Ingeniería de Sistemas, Universidad Nacional Tecnológica de Lima Sur (UNTELS) |
| **Curso** | Sistemas Inteligentes (Mg. Raúl E. Huarote Zegarra) |
| **Destino** | Revista científica indexada (peer review) — formato ACM (`acmart`) |
| **Estado** | Diseño aprobado — pendiente escribir plan de implementación |

---

## 1. Resumen ejecutivo

Se construirá **SATEC**, el primer sistema de **alerta temprana de brotes de la enfermedad de Carrión** (bartonelosis humana, *Bartonella bacilliformis*) basado en aprendizaje automático. Usa 25 años de vigilancia epidemiológica nacional del MINSA (2000–2024, ~46.120 casos) **enriquecidos con clima (NASA POWER) y población (INEI)**, para predecir, a nivel de **provincia** y con **4 semanas de anticipación**, si una zona entrará en **nivel de brote** según el **canal endémico** (estándar OPS/MINSA).

Se comparan tres paradigmas de clasificación —**Árbol de Decisión**, **Red Neuronal** y **ensembles** (Random Forest / Gradient Boosting)— contra **baselines epidemiológicos**, bajo **validación temporal** estricta. El sistema se despliega como una **aplicación web nueva e independiente** con un mapa de riesgo del Perú, y como **código y datos reproducibles**.

Este trabajo reemplaza el enfoque previo (clasificación de la *fase* aguda vs. eruptiva, con un conjunto clínico sintético) por un problema con **datos 100% reales** y **utilidad directa en salud pública**.

## 2. Motivación y contribución (la novedad)

La comparación "Red Neuronal vs. Árbol de Decisión" por sí sola no constituye novedad suficiente para una revista indexada. La contribución real de SATEC es:

1. **Primer sistema de alerta temprana de Carrión con ML.** La literatura ha aplicado ML al dengue y a la leishmaniasis en Perú, pero **no a la enfermedad de Carrión** para predicción de brotes.
2. **Integración del canal endémico (herramienta epidemiológica clásica de la OPS) como objetivo de aprendizaje supervisado.** Aporte metodológico replicable a otras enfermedades de vigilancia.
3. **Enriquecimiento multifuente** (vigilancia + clima rezagado + población) sobre una enfermedad transmitida por vector sensible al clima.
4. **Comparación rigurosa de paradigmas** (interpretables vs. caja negra vs. ensembles) bajo **validación temporal**, con baselines que prueban si el ML aporta sobre métodos epidemiológicos estándar.
5. **Sistema abierto, reproducible y desplegado**, usable por personal de salud pública.

## 3. Objetivos

**General.** Desarrollar y validar un sistema de alerta temprana de brotes de la enfermedad de Carrión a nivel provincial en el Perú, comparando árboles de decisión, redes neuronales y ensembles sobre datos de vigilancia enriquecidos con clima y población.

**Específicos.**
- O1. Construir un pipeline reproducible que transforme la vigilancia de casos individuales del MINSA en una serie espacio-temporal provincia × semana, con ejemplos negativos imputados.
- O2. Definir el objetivo de "brote" mediante el canal endémico, respetando la causalidad temporal.
- O3. Enriquecer con variables climáticas rezagadas (NASA POWER) y poblacionales (INEI).
- O4. Entrenar y comparar AD, RN y ensembles, frente a baselines (canal endémico clásico y persistencia), bajo validación temporal.
- O5. Evaluar con métricas adecuadas a eventos raros (recall de brotes, AUC-PR), calibración e interpretabilidad (SHAP).
- O6. Desplegar una aplicación web con mapa de riesgo y comparador de modelos.
- O7. Redactar el artículo en formato ACM para una revista indexada.

## 4. Alcance

**Incluye:**
- Predicción de **nivel de alerta de brote** (clasificación) a 4 semanas, a nivel **provincial**, en zonas endémicas.
- Datos reales: MINSA + NASA POWER + INEI.
- Modelos: AD, RN, RF, Gradient Boosting; baselines: canal endémico y persistencia.
- Validación temporal, métricas, calibración, interpretabilidad.
- Web nueva e independiente + paper ACM.

**No incluye (YAGNI / fuera de alcance):**
- Detección/diagnóstico clínico en pacientes individuales (no hay datos clínicos reales públicos; descartado en brainstorming).
- Clasificación de fase (aguda vs. eruptiva) como objetivo central — puede aparecer como variable o análisis secundario, no como foco.
- Modelos secuenciales profundos (LSTM/Transformers) — fuera de alcance para mantener el eje AD/RN y evitar *scope creep*. Se mencionan como trabajo futuro.
- Predicción a nivel distrital (demasiado ruido) — trabajo futuro.

## 5. Formulación del problema de ML

**Clasificación supervisada de nivel de alerta.** Para cada par (provincia *p*, semana *t*), predecir si en la ventana **[t+1, t+4]** la provincia alcanzará **nivel de brote** según su canal endémico.

- **Objetivo principal:** binario — `brote` (los casos superan el umbral de epidemia, ≥ Q3 del canal endémico, en las próximas 4 semanas) vs. `no-brote`.
- **Objetivo secundario (opcional):** multiclase por zonas del canal (seguridad / alarma / epidemia).

Alternativas consideradas y descartadas como foco: regresión de incidencia (complementaria) y forecasting secuencial (fuera de alcance). La clasificación encaja directamente con el eje AD vs. RN del trabajo.

## 6. Decisiones de diseño clave

| Decisión | Elección | Justificación |
|---|---|---|
| Granularidad espacial | **Provincia** | Equilibrio señal/ruido; las zonas endémicas se ubican bien a nivel provincial. |
| Granularidad temporal | **Semana epidemiológica** | Estándar de vigilancia; ya presente en los datos. |
| Horizonte de predicción | **4 semanas** | Margen real de acción para salud pública; suaviza ruido semanal. |
| Definición de objetivo | **Canal endémico (OPS)** | Estándar epidemiológico; aporta rigor y novedad metodológica. |
| Tipo de tarea | **Clasificación binaria** (+ multiclase opcional) | Encaja con AD vs. RN; sensible a eventos raros. |
| Validación | **Temporal (rolling-origin)** | Evita fuga de información del futuro; realista. |
| Zonas | **Endémicas** (con historia de transmisión) | Evita diluir la señal con provincias sin casos. |

## 7. Datos

### 7.1 Fuente primaria — MINSA (real, ya disponible)
- Vigilancia epidemiológica de la enfermedad de Carrión, 2000–2024, ~46.120 registros.
- Columnas: `departamento, provincia, distrito, localidad, enfermedad, ano, semana, diagnostic, diresa, ubigeo, localcod, edad, tipo_edad, sexo`.
- Cada fila = un caso confirmado (no hay ceros ni controles sanos → se imputan en el pipeline).
- Fuente: Plataforma Nacional de Datos Abiertos (datosabiertos.gob.pe).

### 7.2 Clima — NASA POWER (a verificar)
- Variables: precipitación, temperatura (media/min/max), humedad relativa, por coordenadas.
- Resolución diaria → agregada a semana epidemiológica.
- Se asignan coordenadas representativas por provincia (centroide).
- Se incorporan **rezagos** (p. ej. 2–8 semanas) por la respuesta diferida del vector *Lutzomyia*.

### 7.3 Población — INEI (a verificar)
- Proyecciones de población por provincia/año → cálculo de **tasas por 100.000 hab.**
- Evita que provincias grandes dominen los conteos absolutos.

### 7.4 Verificaciones pendientes (antes de comprometer en implementación)
- [ ] Confirmar cobertura temporal y formato de NASA POWER para los años 2000–2024.
- [ ] Confirmar disponibilidad de proyecciones de población provincial del INEI para todo el rango.
- [ ] Conseguir **GeoJSON de provincias del Perú** (mapa web + centroides para clima).
- [ ] Verificar consistencia de nombres/ubigeo de provincias entre fuentes.

## 8. Pipeline de datos

1. **Carga y limpieza** del CSV MINSA (normalizar texto de provincia, validar ubigeo, convertir edad a años, mapear fase).
2. **Agregación** → casos por (provincia × año × semana epidemiológica). Conteo total y, opcionalmente, por fase.
3. **Rejilla completa** → producto cartesiano (provincias endémicas × años × semanas 1–53), imputando **0 casos** donde no hubo notificación. Esto genera los ejemplos `no-brote`.
4. **Filtro de zonas endémicas** → provincias con ≥ N casos históricos (umbral a fijar; documentado).
5. **Enriquecimiento** → unir clima rezagado (NASA POWER) y población (INEI) por provincia-semana/año.
6. **Cálculo de tasas** por 100.000 hab.
7. **Construcción del canal endémico** por provincia (ver §9), respetando causalidad temporal.
8. **Etiquetado del objetivo** (brote en [t+1, t+4]).
9. **Ingeniería de características** (ver §10).
10. **Particionado temporal** (ver §12) y exportación del dataset modelable a `data/processed/`.

## 9. Definición del objetivo: canal endémico

El **canal endémico** (OPS) resume el comportamiento histórico esperado de casos por semana del año:

- Para cada semana epidemiológica *s* y provincia *p*, se calcula una medida de tendencia central (mediana o media geométrica) y bandas (cuartiles Q1, Q3) sobre los **años previos** disponibles.
- Zonas: **éxito** (< Q1), **seguridad** (Q1–mediana), **alarma** (mediana–Q3), **epidemia/brote** (> Q3).
- **Objetivo de ML:** una provincia-semana se etiqueta `brote` si en la ventana [t+1, t+4] los casos observados superan el **umbral de epidemia** (≥ Q3, por encima de la zona de alarma) de su canal endémico.
- **Causalidad temporal:** el canal de referencia se construye únicamente con datos **anteriores** al punto de predicción (ventana móvil de años), nunca con el futuro, para no contaminar la validación.

## 10. Ingeniería de características

- **Autorregresivas:** casos/tasa en t, t-1, t-2, t-4; media móvil (4 y 8 semanas); tendencia reciente.
- **Estacionales:** seno y coseno de la semana epidemiológica; mes.
- **Climáticas:** precipitación/temperatura/humedad con rezagos (2–8 semanas) y medias móviles.
- **Estructurales:** provincia (codificada), departamento, año, altitud (opcional), población.
- **Derivadas del canal endémico:** posición actual respecto a su propio canal (z-score / zona actual).

## 11. Modelos

| Modelo | Librería | Rol |
|---|---|---|
| Árbol de Decisión (con y sin poda) | scikit-learn | Eje del trabajo · interpretable · evidencia de sobreajuste |
| Red Neuronal feedforward | tf_keras / TensorFlow | Eje del trabajo · generalización estable |
| Random Forest | scikit-learn | Ensemble de referencia (bagging) |
| Gradient Boosting (XGBoost / LightGBM) | xgboost / lightgbm | Ensemble de referencia (boosting) · estado del arte tabular |
| **Canal endémico clásico** | propio | **Baseline epidemiológico** |
| **Persistencia** (predice = estado actual) | propio | **Baseline ingenuo** |

La RN usa normalización min-max (parámetros guardados para la web); los árboles/ensembles no la requieren. Los baselines son críticos: si el ML no supera al canal endémico clásico, ese también es un resultado honesto y publicable.

## 12. Validación y métricas

- **Esquema:** validación **temporal rolling-origin** (p. ej. entrenar 2000–2018, validar 2019; luego 2000–2019 → 2020; etc.). Reporte agregado. **Nunca** partición aleatoria.
- **Métricas primarias** (eventos raros): **Recall/sensibilidad de brotes**, **AUC-PR**, F1.
- **Métricas secundarias:** AUC-ROC, precisión, especificidad, exactitud, matriz de confusión.
- **Calibración:** curva de fiabilidad + **Brier score**.
- **Interpretabilidad:** importancia de variables y **SHAP** (qué impulsa las alertas).
- **Brecha de generalización** (train vs. test) para evidenciar sobreajuste, en continuidad con el paper previo.

## 13. Manejo del desbalance

Los brotes son raros (clase minoritaria). Estrategias a evaluar y documentar:
- Ponderación de clases (`class_weight`).
- Ajuste del umbral de decisión optimizando recall/AUC-PR.
- Remuestreo (SMOTE u *oversampling*) solo dentro del pliegue de entrenamiento (sin fuga).

## 14. Aplicación web (nueva e independiente)

- **Independiente** de la web actual (Titanic/Carrión). Proyecto y diseño visual propios.
- **Componentes:**
  - 🗺️ **Mapa de riesgo del Perú** (coroplético por provincia) con el nivel de alerta predicho.
  - Selector provincia + semana → predicción de AD, RN y ensemble con confianza.
  - Visualización del **canal endémico** por provincia (curva clásica).
  - Panel comparativo de modelos (métricas).
- **Inferencia:** RN vía TensorFlow.js; árboles vía recorrido JSON; o predicciones precomputadas servidas estáticamente.
- **Stack:** React + TF.js (CDN) + librería de mapas (GeoJSON de provincias). Despliegue estático (Vercel).
- **Diseño visual:** profesional, tema propio de salud pública; se desarrollará con un skill de diseño frontend en la fase de implementación.

## 15. El artículo (paper)

- **Formato:** ACM (`acmart`), español con título y abstract en inglés y español (multi-idioma permitido por el template).
- **Reaprovecha** del paper actual: introducción, parte de la revisión de literatura, marco teórico de RN/AD, ecuaciones.
- **Reescribe:** materiales y métodos (datos, canal endémico, features, modelos, validación), resultados, discusión, conclusiones.
- **Añade** (exigencias de revista indexada): sección de **Trabajos relacionados**, **Limitaciones**, **Disponibilidad de datos y código**, declaración de **conflicto de interés**.
- **Figuras nuevas:** mapa de riesgo, canal endémico, curvas PR/ROC, calibración, SHAP, matrices de confusión, importancia de variables.

## 16. Estructura del proyecto

```
SATEC-Carrion/
├── README.md
├── docs/superpowers/specs/2026-06-18-satec-carrion-alerta-brotes-design.md
├── data/
│   ├── raw/         # MINSA, clima, población, geojson
│   ├── interim/     # intermedios
│   └── processed/   # dataset modelable (provincia-semana + features + target)
├── src/
│   ├── data/        # limpieza, agregación, rejilla, canal endémico
│   ├── features/    # ingeniería de características
│   ├── models/      # entrenamiento AD, RN, ensembles, baselines
│   ├── evaluation/  # validación temporal, métricas, calibración, SHAP
│   └── export/      # exportar a TF.js / JSON para la web
├── notebooks/       # exploración y generación de figuras
├── models/          # modelos entrenados
├── results/         # métricas y figuras
├── web/             # aplicación web nueva
└── paper/           # artículo ACM (docx) y figuras
```

## 17. Stack tecnológico

- **Datos/ML:** Python 3.12, pandas, numpy, scikit-learn, tf_keras/TensorFlow, xgboost/lightgbm, shap, matplotlib.
- **Clima:** API NASA POWER. **Población:** INEI. **Mapa:** GeoJSON de provincias del Perú.
- **Web:** React + TensorFlow.js + librería de mapas; despliegue estático (Vercel).
- **Reproducibilidad:** semillas fijas, `requirements.txt`, scripts ejecutables de punta a punta.

## 18. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Señal predictiva débil (Carrión es local/esporádico) | Baselines lo revelan; caracterizar la dificultad es un resultado honesto y publicable. |
| Clima/población no disponibles a la granularidad deseada | Verificar antes de comprometer (§7.4); plan B: variante "autosuficiente" solo-MINSA. |
| Inconsistencia de nombres/ubigeo entre fuentes | Tabla de homologación por ubigeo; validación manual. |
| Subnotificación y cambios de definición de caso (25 años) | Declarar como limitación; análisis de sensibilidad por período. |
| Desbalance extremo de clases | Ponderación, ajuste de umbral, remuestreo sin fuga (§13). |
| *Scope creep* (web + ML + paper) | Fases con entregables; web e informe se abordan tras validar el modelo. |

## 19. Limitaciones (a declarar en el paper)

- Datos de **vigilancia pasiva**: subnotificación y sesgos de acceso a salud.
- El canal endémico depende de la calidad histórica de la serie.
- Variables climáticas asignadas por centroide provincial (no captan microclimas).
- Sin validación clínica individual (no es objetivo del sistema).

## 20. Criterios de éxito

- Pipeline reproducible de extremo a extremo (raw → dataset modelable → modelos → métricas → web).
- Comparación AD/RN/ensembles vs. baselines bajo validación temporal, con métricas adecuadas a eventos raros.
- Conclusión defendible (el ML supera —o no— al canal endémico clásico, con evidencia).
- Web funcional con mapa de riesgo y comparador.
- Artículo ACM completo, honesto y listo para someter a una revista indexada.

## 21. Trabajo futuro

- Granularidad distrital con modelos espaciales.
- Modelos secuenciales (LSTM/Transformers temporales).
- Integración con datos entomológicos y de movilidad.
- Validación prospectiva en campo con una DIRESA.

## 22. Referencias clave (semilla)

- Maguiña Vargas, C. (2009). Bartonelosis o enfermedad de Carrión. *Acta Médica Peruana*.
- OPS. Bartonelosis (enfermedad de Carrión) / metodología del canal endémico.
- Rufasto-Goche et al. (2025). Dinámica espacio-temporal del dengue en Perú (datos MINSA). *PLOS One*.
- Vadmal et al. (2023). Predicción de vectores de *Leishmania* en las Américas. *PLOS NTD*.
- Grinsztajn et al. (2022); Shwartz-Ziv & Armon (2022). Modelos de árbol vs. deep learning en datos tabulares.
- MINSA. Datos abiertos de vigilancia de la enfermedad de Carrión, 2000–2024.
```
