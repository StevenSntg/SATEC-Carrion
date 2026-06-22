# Corrección de modelos y artículo SATEC: F1, validación de origen móvil, ensembles y observaciones del profesor

- **Fecha:** 2026-06-22
- **Estado:** Diseño aprobado (pendiente de revisión del spec) → siguiente paso: plan de implementación
- **Ámbito:** `src/satec/models/`, `src/satec/web_export/`, `notebooks/`, `paper/`, `results/`, `requirements.txt`

## 1. Contexto y problema

El artículo SATEC compara un Árbol de Decisión y una Red Neuronal para la alerta
temprana de brotes de la enfermedad de Carrión. El profesor revisó el `.docx` y
dejó **19 observaciones** (texto rojo/mayúsculas), y de forma independiente se
detectó que el **F1 reportado es muy bajo** (RN 0,39; árboles 0,13–0,17).

## 2. Diagnóstico del F1 bajo

El F1 bajo **no se debe a modelos malos** (la RN tiene AUC-ROC 0,983 y AUC-PR
0,745). Tiene tres causas, en orden de importancia:

1. **Colapso de prevalencia entre entrenamiento y prueba.** El split actual
   entrena con ≤2019 (prevalencia 10,0 %, 5.398 brotes / 53.985) y prueba con
   2020–2024 (prevalencia **0,27 %**, 42 brotes / 15.616). Los brotes del test
   por año son: 2020→4, **2021→0**, 2022→4, 2023→2, 2024→32. El período de prueba
   coincide con la **subnotificación pandémica de COVID-19** (la vigilancia de
   enfermedades endémicas se desplomó: 0 brotes en 61 provincias en 2021 es la
   firma del colapso de registro, no de una mejora sanitaria). Con 42 positivos,
   la precisión —y por tanto el F1— es matemáticamente baja aunque el modelo
   ordene bien el riesgo.
2. **Umbral fijo en 0,5, nunca optimizado** (`evaluate.py:40`). El F1 depende del
   umbral; con uno elegido en validación sube de forma legítima.
3. **F1 con umbral 0,5 es la métrica inadecuada** para 0,27 % de prevalencia. El
   paper ya usa AUC-PR/sensibilidad como primarias, pero el F1 crudo se exhibe en
   tablas/figuras y "se ve mal".

> El árbol **sin poda** sí es un mal modelo (sobreajuste: train_acc 0,999,
> AUC-PR 0,091); se conserva como ejemplo didáctico de sobreajuste.

## 3. Objetivos y criterios de éxito

1. **F1 representativo y legítimo**: medido bajo validación de origen móvil con
   umbral óptimo, no inflado artificialmente.
2. **Resolver las 19 observaciones del profesor** (ver §6).
3. **Reproducibilidad de extremo a extremo**: reentrenar → regenerar
   `metricas_modelos.csv` → regenerar figuras → regenerar `.md` → regenerar
   `.docx` ACM, todo con scripts versionados.
4. **Consistencia sistema↔paper**: la web usa los mismos modelos reportados.

**Criterios de éxito verificables:**
- `pytest` en verde tras los cambios (suite existente + nuevas pruebas).
- `results/metricas_modelos.csv` con 6 modelos y columnas de F1 en umbral óptimo.
- `paper/SATEC_articulo_ACM.docx` regenerado sin observaciones pendientes.
- Citas [1]…[N] en orden de aparición (linealidad IEEE/ACM).

## 4. Decisiones tomadas (con el usuario)

| # | Decisión | Elección |
|---|---|---|
| 1 | Cómo abordar el F1 | **Validación de origen móvil + umbral óptimo** |
| 2 | Modelos | **Añadir Random Forest y Gradient Boosting** |
| 3 | Alcance del encargo | **Plan detallado primero, luego ejecutar** |
| 4 | Comparación con literatura | **Buscar cifras en la web y validar con el usuario** |
| 5 | Web | **Re-exportar modelos para mantener consistencia** |

## 5. Diseño detallado

### 5.1 Modelos nuevos (`src/satec/models/train.py`)
- `train_random_forest(X, y, ...)`: `RandomForestClassifier`
  (`class_weight="balanced"`, `n_estimators≈300`, `random_state=42`).
- `train_gradient_boosting(X, y, ...)`: `HistGradientBoostingClassifier`
  (soporta `class_weight="balanced"`); alternativa `GradientBoostingClassifier`
  con `sample_weight` si hiciera falta.
- Árbol y RN se conservan tal cual.

### 5.2 Validación de origen móvil (`src/satec/models/rolling_origin.py`, nuevo)
- Esquema causal por año objetivo Y (rango propuesto **2016–2024**):
  - `train_inner = años ≤ Y-2`, `val = últimos años antes de Y` (ventana de Y-1
    hacia atrás hasta reunir un mínimo de positivos), `test = año Y`.
  - Entrenar en `train_inner`; elegir umbral `t*` que **maximiza F1 en `val`**;
    predecir `Y` con `t*`. El año de prueba nunca interviene en la elección del
    umbral (sin fuga).
  - **Caso borde (validación sin brotes, p. ej. 2021):** si la ventana de
    validación no reúne un mínimo de positivos (umbral configurable, p. ej. ≥5),
    se amplía hacia atrás; si aun así no los hay, se usa un umbral de respaldo
    (el de la prevalencia de `train_inner` o el último `t*` válido). El caso se
    registra en un log para trazabilidad.
- **Métricas independientes de umbral** (AUC-PR, AUC-ROC, Brier): sobre el *pool*
  de predicciones de todos los años de prueba.
- **Métricas dependientes de umbral** (Precisión, Sensibilidad, F1, F2,
  Especificidad): agregando la matriz de confusión total (suma de VP/FP/FN/VN de
  todos los cortes con su `t*`).
- El canal endémico y el target ya son causales en
  `dataset_enriched.parquet`; no se recalculan por corte.
- Se mantiene además el corte único ≤2019 / 2020–2024 como **estudio de robustez**
  (para mostrar explícitamente el efecto de la subnotificación pandémica).

### 5.3 Métricas y resultados (`results/`)
- Nuevo `metricas_modelos.csv` con 6 modelos (RN, RF, GB, árbol-8, árbol sin
  poda, baseline canal endémico) y columnas: precisión, sensibilidad, F1, **F2**,
  especificidad, AUC-PR, AUC-ROC, Brier, umbral `t*`, VP/FP/FN/VN.
- `metrics.py` extendido con F2 y especificidad.

### 5.4 Figuras (`src/satec/models/figures*.py`)
- **Arquitectura de la RN** (nueva figura): diagrama 24→32→32→1, ReLU/sigmoide,
  normalización min-max y entropía cruzada (resuelve p115).
- **Etiquetas numéricas** en todas las barras (resuelve p124).
- **Matrices de confusión de RN y RF** con conteos **y porcentajes** y leyenda
  clara de VP/FP/FN/VN (resuelve p111, p116, p120–p122).
- **Calibración** regenerada (más estable por el mayor número de positivos del
  origen móvil; resuelve p126).
- Todas las figuras propias llevan **"Fuente: SATEC [N]"** (resuelve p147).

### 5.5 Texto del paper (`paper/SATEC_articulo_ACM.md`)
- **§2.7** reescrita a origen móvil; **nueva subsección** de RF y GB.
- **Tabla 1** ampliada a 6 modelos con F1 en umbral óptimo y F2.
- **§3.2 Matriz de confusión** reescrita: VP/FP/FN/VN como *provincias-semana
  correctamente o incorrectamente alertadas*, con % de error por modelo. Se
  aclara de forma explícita que el sistema **predice brote por provincia-semana,
  no diagnostica personas**, conectando con la intuición del profesor sin afirmar
  algo incorrecto (resuelve p116, p118, p119, p120, p121, p122).
- **Discusión**: tabla/párrafo de **comparación cuantitativa con la literatura**
  (dengue de Rufasto-Goche, Leishmania de Vadmal, nicho de *Lutzomyia*, etc.),
  con cifras buscadas en la web y validadas (resuelve p151, p156).
- **Limitaciones**: la "partición única" se elimina; se añade el efecto de la
  subnotificación pandémica. **Agradecimientos** sin repetir la institución
  (resuelve p159).
- **Abstract/Resumen** actualizados a los números nuevos.

### 5.6 Referencias y formato (`paper/SATEC_articulo_ACM.md`, `paper/build_acm.py`)
- **Renumerar las referencias en orden de aparición** (linealidad IEEE/ACM) y
  reordenar todas las citas del cuerpo en consecuencia (resuelve p113, p12/p14/p15).
- **Fuente de datos** (MINSA, datosabiertos.gob.pe) citada en referencias y
  mencionada en el texto donde se describe la fuente (resuelve p123).
- **Palabras clave dentro de los resúmenes** (resuelve p112).
- **Sin sangría** en el estilo de párrafo del generador (resuelve p127).

### 5.7 Notebooks (`notebooks/build_notebooks.py`)
- Nuevo `03_random_forest_train_test.ipynb` (con celdas de Gradient Boosting).
- 01 y 02 actualizados: comentario de prevalencia corregido (no "~8 %"), umbral
  óptimo y una sección breve de validación de origen móvil.

### 5.8 Web (`src/satec/web_export/`)
- Re-exportar árbol-8 + RN tras el reentrenamiento (consistencia con el paper).
- RF en la web es **opcional** (serializar un bosque a JSON es voluminoso); se
  evaluará en el plan si se incluye o se deja como trabajo de la web.

### 5.9 Reproducibilidad (`requirements.txt`)
- Añadir `scikit-learn`, `tf-keras` (y `tensorflow`/`tensorflowjs` para export),
  hoy ausentes pese a usarse.

## 6. Mapeo de las 19 observaciones del profesor → acción

| Nota | Observación (resumen) | Acción | §  |
|---|---|---|---|
| p111 | Figuras matplotlib de RN **y Random Forest** | Añadir RF + figuras | 5.1, 5.4 |
| p112 | Palabras clave dentro de los resúmenes | Keywords en abstract/resumen | 5.6 |
| p113 | Linealidad de citas [1]…[N] | Renumerar en orden de aparición | 5.6 |
| p114 | Pasar a PDF para QA→PRD | Nota de proceso (build final) | 5.5 |
| p115 | Explicar estructura del modelo en la figura | Figura de arquitectura RN | 5.4 |
| p116 | Balance VN/VP | Matriz con % y explicación | 5.4, 5.5 |
| p118 | Especificar diagnosticados/no, padecen/no | Reescritura §3.2 | 5.5 |
| p119 | Enfocar en quienes sí padecen | Énfasis en sensibilidad | 5.5 |
| p120/p121 | Explicar los 98 FP de la RN | Texto + figura | 5.5 |
| p122 | FP del árbol-8 (389) y % de error | Texto + figura | 5.5 |
| p123 | Link de la fuente en referencias | Referencia + mención | 5.6 |
| p124 | Etiquetas de datos con números | Data labels en figuras | 5.4 |
| p126 | Arreglar la calibración | Regenerar calibración | 5.4 |
| p127 | Sin sangría | Estilo de párrafo | 5.6 |
| p147 | "Fuente: SATEC" en figuras (IEEE) | Pie de figura con cita | 5.4, 5.6 |
| p151 | Comparación cuantitativa con literatura | Tabla comparativa | 5.5 |
| p156 | Solo los modelos en la comparación | Curar la comparación | 5.5 |
| p159 | No repetir la institución | Editar agradecimientos | 5.5 |
| p12/p14/p15 | Marcas en citas desordenadas | Cubierto por p113 | 5.6 |

## 7. Fuera de alcance
- Cambiar la definición del target o del canal endémico.
- Descenso a granularidad distrital o nuevas fuentes de datos.
- Rediseño de la UX de la web (solo re-exportación de modelos).

## 8. Riesgos y mitigaciones
- **Reproducibilidad de la RN** (TensorFlow/hardware): fijar semilla; reportar que
  los valores pueden variar levemente.
- **Renumeración de referencias**: riesgo de romper la correspondencia
  cita↔referencia; mitigar verificando que toda cita del cuerpo exista y esté en
  orden creciente de primera aparición.
- **Cifras de literatura**: deben validarse con el usuario antes de publicarlas.
- **Regeneración del `.docx`**: depende de la plantilla ACM y del XSLT de Office
  (ambos verificados como presentes).

## 9. Verificación final
- `pytest -q` en verde.
- Revisión visual del `.docx`: 6 modelos en Tabla 1, matrices con %, figura de
  arquitectura, citas en orden, keywords en resúmenes, sin sangría, fuentes en
  figuras, agradecimientos corregidos.
- Las 19 notas marcadas como resueltas en este documento.
