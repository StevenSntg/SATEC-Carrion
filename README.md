# SATEC — Sistema de Alerta Temprana de la Enfermedad de Carrión

Sistema de **alerta temprana de brotes de la enfermedad de Carrión** (bartonelosis humana,
*Bartonella bacilliformis*) basado en aprendizaje automático, construido sobre 25 años de
vigilancia epidemiológica del MINSA (2000–2024) enriquecidos con datos de **clima** (NASA POWER)
y **población** (INEI).

> **Estado:** en desarrollo · Diseño aprobado · Pendiente plan de implementación.

## Objetivo

Predecir, a nivel de **provincia** y con **4 semanas de anticipación**, si una zona entrará en
**nivel de brote** según el **canal endémico** (estándar OPS/MINSA), comparando **Árboles de
Decisión**, **Redes Neuronales** y **ensembles** frente a baselines epidemiológicos, bajo
**validación temporal**.

A diferencia del trabajo previo (clasificación de la *fase* aguda vs. eruptiva con datos
sintéticos), SATEC usa **datos 100 % reales** y tiene **utilidad directa en salud pública**.

## Documento de diseño

La especificación completa del proyecto está en
[`docs/superpowers/specs/2026-06-18-satec-carrion-alerta-brotes-design.md`](docs/superpowers/specs/2026-06-18-satec-carrion-alerta-brotes-design.md).

## Estructura del proyecto

```
SATEC-Carrion/
├── docs/        # Diseño (spec) y documentación
├── data/        # raw (fuentes) · interim · processed
├── src/         # data · features · models · evaluation · export
├── notebooks/   # exploración y figuras
├── models/      # modelos entrenados
├── results/     # métricas y figuras
├── web/         # aplicación web (mapa de riesgo)
└── paper/       # artículo ACM y figuras
```

## Autores

- Jaqueline Alvarez Rocca
- Carlos Meza Pelaez
- Carlos Steven Santiago Flores

Escuela Profesional de Ingeniería de Sistemas, Universidad Nacional Tecnológica de Lima Sur
(UNTELS), Lima, Perú. Curso de Sistemas Inteligentes.

## Fuentes de datos

- **MINSA** — Vigilancia epidemiológica de la enfermedad de Carrión, 2000–2024
  (Plataforma Nacional de Datos Abiertos, datosabiertos.gob.pe).
- **NASA POWER** — variables climáticas.
- **INEI** — proyecciones de población.
