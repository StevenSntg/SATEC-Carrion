# Alerta temprana de brotes de la enfermedad de Carrión en el Perú: redes neuronales frente a árboles de decisión sobre datos de vigilancia y clima

*Título corto (encabezado): Redes neuronales y árboles de decisión para la alerta de brotes de Carrión*

**Jaqueline Alvarez Rocca**, Escuela Profesional de Ingeniería de Sistemas, Universidad Nacional Tecnológica de Lima Sur (UNTELS), Lima, Perú.
**Carlos Meza Pelaez**, Escuela Profesional de Ingeniería de Sistemas, UNTELS, Lima, Perú.
**Carlos Steven Santiago Flores**, Escuela Profesional de Ingeniería de Sistemas, UNTELS, Lima, Perú.

---

## Abstract

Carrión's disease (human bartonellosis), caused by *Bartonella bacilliformis* and transmitted by the *Lutzomyia* sand fly, is a neglected disease endemic to the Peruvian inter-Andean valleys, with a potentially lethal acute phase. We present **SATEC**, an early-warning system that predicts, at the province level and four weeks ahead, whether a zone will enter an **outbreak state** as defined by the **endemic channel** (the standard surveillance tool of PAHO/MINSA). The system is built on 25 years of national open surveillance data from the Peruvian Ministry of Health (MINSA, 2000–2024; ~46,120 case records), aggregated into a province × epidemiological-week panel with imputed zero weeks, and enriched with **climate** variables (NASA POWER: precipitation, temperature, humidity, with lags) and **population** (2017 census, for incidence rates). On weakly informative variables, we compare a feed-forward **Neural Network** (Keras) against a **Decision Tree** (scikit-learn) under **strict temporal validation** (train ≤2019, test 2020–2024), reporting metrics suited to rare events (recall, AUC-PR) plus calibration and permutation importance. The neural network attains the best balance (recall 0.81, AUC-PR 0.745, Brier 0.012) and outperforms both the unpruned tree—which overfits—and the classical endemic-channel baseline; the recent moving average of cases dominates, while climate and population add real signal. The system is deployed as an open, reproducible web application with a provincial risk map. We conclude that machine learning can add value over classical surveillance for a neglected disease, provided the variables are informative and the evaluation respects temporal causality.

**CCS Concepts:** • Computing methodologies → Machine learning; Neural networks; Classification and regression trees. • Applied computing → Health informatics; Life and medical sciences.

**Keywords:** enfermedad de Carrión; bartonelosis; alerta temprana; redes neuronales; árboles de decisión; canal endémico; vigilancia epidemiológica; aprendizaje automático; Perú.

## Resumen

La enfermedad de Carrión (bartonelosis humana), causada por *Bartonella bacilliformis* y transmitida por el vector *Lutzomyia*, es una enfermedad desatendida y endémica de los valles interandinos del Perú, con una fase aguda potencialmente letal. Presentamos **SATEC**, un sistema de alerta temprana que predice, a nivel de **provincia** y con **cuatro semanas de anticipación**, si una zona entrará en **estado de brote** según el **canal endémico** (la herramienta estándar de vigilancia de la OPS/MINSA). El sistema se construye sobre 25 años de datos abiertos de vigilancia del Ministerio de Salud (MINSA, 2000–2024; ~46.120 registros de casos), agregados en un panel provincia × semana epidemiológica con imputación de semanas en cero, y enriquecidos con variables **climáticas** (NASA POWER: precipitación, temperatura, humedad, con rezagos) y de **población** (censo 2017, para tasas de incidencia). Se comparan una **Red Neuronal** prealimentada (Keras) y un **Árbol de Decisión** (scikit-learn) bajo **validación temporal estricta** (entrenamiento ≤2019, prueba 2020–2024), reportando métricas adecuadas a eventos raros (sensibilidad, AUC-PR) además de calibración e importancia por permutación. La red neuronal logra el mejor equilibrio (sensibilidad 0,81; AUC-PR 0,745; Brier 0,012) y supera tanto al árbol sin poda —que sobreajusta— como al baseline clásico del canal endémico; la media móvil reciente de casos domina, mientras que el clima y la población aportan señal real. El sistema se despliega como una aplicación web abierta y reproducible con un mapa de riesgo provincial. Concluimos que el aprendizaje automático puede aportar valor sobre la vigilancia clásica en una enfermedad desatendida, siempre que las variables sean informativas y la evaluación respete la causalidad temporal.

---

## 1. Introducción

La enfermedad de Carrión, o bartonelosis humana, es una enfermedad bacteriana desatendida, endémica de los valles interandinos del Perú, causada por *Bartonella bacilliformis* y transmitida por flebótomos del género *Lutzomyia* [1], [17]. Lleva el nombre de Daniel Alcides Carrión, mártir de la medicina peruana. Presenta una fase aguda (anemia hemolítica severa, «Fiebre de la Oroya»), de alta letalidad sin tratamiento, y una fase eruptiva («Verruga Peruana»), más benigna. Su carácter focal y estacional, ligado a las condiciones ambientales del vector, la convierte en candidata natural para sistemas de **alerta temprana** que anticipen la intensificación de la transmisión y orienten la respuesta de salud pública.

El aprendizaje automático se ha consolidado como herramienta de apoyo a la decisión en salud pública, con dos grandes familias de modelos supervisados: las redes neuronales artificiales, que aproximan funciones complejas mediante combinaciones no lineales de sus entradas [3], [4], y los árboles de decisión, que clasifican mediante reglas interpretables [5], [6]. Ambos resuelven el mismo problema de clasificación, pero difieren en generalización, interpretabilidad y sensibilidad a la calidad de los datos. En datos tabulares —el formato típico de los registros epidemiológicos— la evidencia reciente muestra que los modelos basados en árboles siguen siendo muy competitivos frente a las redes profundas [11], [12], lo que motiva una comparación controlada sobre datos reales.

Este trabajo presenta **SATEC** (Sistema de Alerta Temprana de la Enfermedad de Carrión). A diferencia de aproximaciones que clasifican la fase clínica sobre conjuntos sintéticos, SATEC aborda un problema con **utilidad directa y datos enteramente reales**: predecir, por provincia y con cuatro semanas de anticipación, si una zona entrará en estado de brote. La contribución es cuádruple. Primero, se transforma la vigilancia de casos individuales del MINSA en un panel espacio-temporal con imputación de semanas sin casos, y se define el objetivo mediante el **canal endémico**, integrando una herramienta epidemiológica clásica como etiqueta de aprendizaje supervisado. Segundo, se enriquecen los datos con clima (NASA POWER) y población (censo 2017). Tercero, se comparan red neuronal y árbol de decisión bajo **validación temporal estricta**, frente a un baseline epidemiológico, con métricas adecuadas a eventos raros, calibración e interpretabilidad. Cuarto, el sistema se despliega como una aplicación web abierta y reproducible. El resto del artículo describe los materiales y métodos (Sección 3), los resultados (Sección 4), su discusión (Sección 5), las limitaciones (Sección 6) y las conclusiones (Sección 7).

## 2. Trabajos relacionados

La aplicación de inteligencia artificial a las enfermedades transmitidas por vectores es un campo en crecimiento. Jiménez-Vásquez et al. [18] emplean análisis in-silico para predecir epítopos de células B en proteínas de *B. bacilliformis*, con el fin de mejorar el diagnóstico serológico de la enfermedad de Carrión. Vadmal et al. [19] entrenan modelos de aprendizaje automático para predecir vectores potenciales de *Leishmania* en América —vectores del género *Lutzomyia*, emparentados con el de Carrión—, identificando focos en Madre de Dios, Perú. Nayak et al. [20] revisan cómo la IA y el aprendizaje profundo transforman el control de las enfermedades vectoriales, desde la detección del vector hasta la predicción de brotes. En el contexto peruano, Rufasto-Goche et al. [21] modelan la dinámica espacio-temporal del dengue con veintidós años de vigilancia del MINSA —la misma fuente que aquí se emplea—. Estos antecedentes muestran un interés creciente por aplicar el aprendizaje automático a enfermedades endémicas peruanas, pero, hasta donde conocemos, **ningún trabajo aborda la alerta temprana de brotes de la enfermedad de Carrión mediante aprendizaje automático**, vacío que este artículo busca atender.

## 3. Materiales y métodos

### 3.1 Datos

**Fuente primaria (MINSA).** Se emplearon los datos abiertos de vigilancia epidemiológica de la enfermedad de Carrión del Ministerio de Salud del Perú [2] (2000–2024; ~46.120 registros de casos confirmados, con departamento, provincia, ubigeo, año, semana epidemiológica, edad, sexo y fase). Cada registro es un caso; la fuente no incluye semanas sin casos, lo que se resuelve en el preprocesamiento.

**Construcción del panel.** Los casos se agregaron por provincia × año × semana epidemiológica (semanas 1–52; la 53 se reasignó a la 52). Se generó la rejilla completa de combinaciones y se imputaron en cero las semanas sin notificación, creando los ejemplos negativos. El análisis se restringió a las **provincias endémicas** (al menos 10 casos históricos en al menos 3 años distintos), obteniéndose **61 provincias** y un panel de **69.601** observaciones provincia-semana.

**Enriquecimiento climático (NASA POWER).** Para el centroide de cada provincia se descargaron series diarias de precipitación, temperatura y humedad relativa de NASA POWER [22], agregadas a semana epidemiológica (precipitación por suma; temperatura y humedad por promedio) y rezagadas (4 y 8 semanas) más medias móviles, en reconocimiento de la respuesta diferida del vector a las condiciones ambientales. La cobertura climática fue del 100 % de las observaciones.

**Población (INEI).** Se incorporó la población provincial del censo 2017 [23] para calcular la **tasa de incidencia por 100.000 habitantes**. Se usa como población de referencia constante (ver Limitaciones).

### 3.2 Definición del objetivo: el canal endémico

El **canal endémico** es la herramienta estándar de la OPS/MINSA para describir el comportamiento esperado de una enfermedad por semana del año [17]. Para cada provincia y semana epidemiológica se calcularon, a partir de los **años previos** disponibles (ventana móvil de hasta cinco años, mínimo tres), los cuartiles de los casos históricos. Una provincia-semana se etiquetó como **brote** si, en la ventana de las **cuatro semanas siguientes**, los casos superaban el tercer cuartil (Q3, zona de epidemia) del canal y eran al menos dos. El canal de referencia se construyó exclusivamente con información anterior al punto de predicción, evitando la fuga de información temporal. La clase resultante está fuertemente desbalanceada (≈7,8 % de brotes).

### 3.3 Características

El vector de entrada (24 variables) combina: términos autorregresivos de casos (valores rezagados y medias móviles de 4 y 8 semanas), estacionalidad (seno y coseno de la semana), variables climáticas y sus rezagos/medias móviles, y la tasa por 100.000 habitantes. Se excluyeron deliberadamente las claves (provincia, año, semana) y el propio canal (Q1, Q2, Q3) para no contaminar el aprendizaje con la definición del objetivo.

### 3.4 Modelos

Se compararon dos paradigmas, ambos con `class_weight` balanceado para el desbalance:

- **Árbol de Decisión** (scikit-learn [7]), criterio de entropía, en dos variantes: sin poda y con profundidad máxima 8.
- **Red Neuronal** prealimentada (Keras [9] sobre TensorFlow [8]): 24 → 32 (ReLU) → 32 (ReLU) → 1 (sigmoide), entropía cruzada binaria, optimizador Adam [10], normalización min-max con parámetros estimados solo en entrenamiento.

Como **referencia epidemiológica** (no un modelo de aprendizaje) se incluyó la **persistencia sobre el canal endémico**: predecir brote futuro si la provincia está hoy por encima de su canal.

### 3.5 Validación y métricas

Se empleó **validación temporal**: entrenamiento con los años ≤2019 y prueba con 2020–2024, sin mezclar años. Dado el desbalance, las métricas primarias son la **sensibilidad (recall) de brotes** y el **área bajo la curva de precisión-exhaustividad (AUC-PR)**; se reportan además precisión, F1, AUC-ROC, exactitud y la matriz de confusión. La **calibración** se evaluó con el *Brier score* y curvas de fiabilidad, y la **interpretabilidad** mediante importancia por permutación (caída en AUC-PR al permutar cada variable).

### 3.6 Sistema y despliegue

El sistema separa un mundo de entrenamiento en Python (que se ejecuta una vez) de un mundo de inferencia en el navegador. La red neuronal se exporta a TensorFlow.js [16] y el árbol a un JSON plano. La aplicación web es una página estática (HTML, CSS, JavaScript con Leaflet) que muestra un **mapa de riesgo coroplético del Perú** por provincia, con el semáforo del canal endémico, y un panel comparativo de los modelos; se despliega de forma estática y reproducible.

## 4. Resultados

### 4.1 Comparación de modelos

La Tabla 1 resume el desempeño sobre el conjunto de prueba (2020–2024). La **red neuronal** logra el mejor equilibrio: la mayor sensibilidad (0,81), el mejor AUC-PR (0,745) y el menor Brier (0,012). El árbol con poda mejora notablemente al árbol sin poda en AUC-PR (0,625 frente a 0,091), y ambos modelos de aprendizaje superan al **baseline del canal endémico** (F1 0,40) en las métricas centradas en brotes. La Figura 1 (`fig_metricas.png`) compara visualmente sensibilidad, AUC-PR y F1 por modelo.

**Tabla 1. Desempeño en el conjunto de prueba temporal (2020–2024).**

| Modelo | Sensibilidad | Precisión | F1 | AUC-PR | AUC-ROC | Brier |
|---|---|---|---|---|---|---|
| Red Neuronal | **0,81** | 0,26 | 0,39 | **0,745** | 0,983 | **0,012** |
| Árbol (poda 8) | 0,71 | 0,07 | 0,13 | 0,625 | 0,871 | 0,015 |
| Árbol (sin poda) | 0,60 | 0,10 | 0,17 | 0,091 | 0,792 | 0,014 |
| Canal endémico (ref.) | 0,64 | 0,29 | 0,40 | — | — | — |

### 4.2 Brecha de generalización

La Figura 2 (`fig_brecha.png`) muestra la brecha entre exactitud de entrenamiento y de prueba. El **árbol sin poda** memoriza el pasado (exactitud de entrenamiento ≈0,999) y se desploma en AUC-PR sobre datos nuevos (0,091): la firma inequívoca del sobreajuste. La poda y, sobre todo, la red neuronal generalizan de forma mucho más estable.

### 4.3 Calibración

La Figura 3 (`fig_calibracion.png`) presenta las curvas de fiabilidad. Los *Brier scores* son bajos (0,012–0,015), favorecidos por la baja prevalencia; la red neuronal presenta la mejor calibración global, aunque con margen de mejora en los deciles de alta probabilidad, donde los brotes son escasos.

### 4.4 Importancia de variables

La Figura 4 (`fig_importancia.png`) reporta la importancia por permutación de la red neuronal. La **media móvil de ocho semanas de casos** es el predictor dominante, seguida del número de casos actuales y de la media móvil de cuatro semanas. De manera relevante para la hipótesis de enriquecimiento, la **tasa por 100.000 habitantes** (población) figura entre las cinco variables más importantes, y la **temperatura** (media móvil) aporta señal apreciable: clima y población contribuyen, más allá de la autocorrelación de casos.

### 4.5 El sistema interactivo

La aplicación web materializa los resultados en un **mapa de riesgo del Perú** donde cada provincia endémica se colorea según su nivel de alerta (bajo/medio/alto) para la semana más reciente; al seleccionar una provincia se muestran la probabilidad de la red neuronal, la decisión del árbol y los casos notificados. El sistema es de uso libre y su código y datos son reproducibles.

## 5. Discusión

Tres conclusiones emergen de los resultados. Primero, **el aprendizaje automático aporta valor sobre la vigilancia clásica**: la red neuronal supera al canal endémico en sensibilidad y AUC-PR, anticipando brotes que la regla clásica no captura. Segundo, **la calidad y la causalidad de las variables son decisivas**: el árbol sin poda alcanza una exactitud engañosamente alta en entrenamiento pero fracasa en datos nuevos, lo que subraya la necesidad de validación temporal y de métricas adecuadas a eventos raros, en lugar de la exactitud. Tercero, **el enriquecimiento es útil**: la población (tasa) y el clima aparecen entre los predictores más importantes, lo que respalda la integración multifuente para una enfermedad sensible a las condiciones ambientales del vector.

En términos de paradigmas, la red neuronal resultó preferible para este problema desbalanceado y ruidoso, mientras que el árbol —especialmente sin poda— ilustra los riesgos del sobreajuste. La elección, sin embargo, no debería guiarse solo por la exactitud, sino también por la interpretabilidad requerida: el árbol ofrece reglas legibles, valiosas para la comunicación con tomadores de decisión.

## 6. Limitaciones

Conviene declarar las limitaciones con franqueza. (i) Los datos son de **vigilancia pasiva**, sujetos a subnotificación y a cambios en la definición de caso a lo largo de 25 años. (ii) La población se tomó del **censo 2017 como referencia constante**, sin capturar la dinámica interanual. (iii) Las variables climáticas se asignaron por **centroide provincial**, sin resolver microclimas. (iv) El sistema **no realiza diagnóstico clínico individual**: es una herramienta de apoyo a la vigilancia y no sustituye la confirmación de laboratorio. (v) Se empleó una única partición temporal; una validación de origen móvil con múltiples cortes fortalecería las conclusiones.

## 7. Conclusiones

Se presentó **SATEC**, el primer sistema de alerta temprana de brotes de la enfermedad de Carrión basado en aprendizaje automático, construido sobre datos reales de vigilancia del MINSA enriquecidos con clima y población, y validado de forma temporalmente honesta. La red neuronal superó al árbol de decisión y al canal endémico clásico en las métricas centradas en brotes; el árbol sin poda evidenció sobreajuste; y el clima y la población aportaron señal predictiva real. El sistema es reproducible y desplegable sin servidor, lo que lo hace útil como instrumento de apoyo a la vigilancia de una enfermedad desatendida. El trabajo futuro incluye la validación de origen móvil, modelos de ensamble (bosques aleatorios, potenciación de gradiente), granularidad distrital, modelos secuenciales y la validación prospectiva en campo con una Dirección Regional de Salud.

## Agradecimientos

A la Escuela Profesional de Ingeniería de Sistemas de la Universidad Nacional Tecnológica de Lima Sur (UNTELS) y, de manera especial, al Mg. Raúl E. Huarote Zegarra, docente del curso de Sistemas Inteligentes, por su orientación. Al MINSA por la publicación de los datos abiertos de vigilancia, y al personal de salud que sostiene la vigilancia de la enfermedad de Carrión.

## Disponibilidad de datos y código

Los datos de vigilancia provienen de los datos abiertos del MINSA [2]; el clima, de NASA POWER [22]; la población, del INEI [23]; los límites provinciales, de un repositorio GeoJSON público. El código del pipeline, los modelos y la aplicación web son reproducibles de extremo a extremo con Python 3.12 y se acompañan de pruebas automatizadas.

## Referencias

[1] C. Maguiña Vargas, «Bartonelosis o enfermedad de Carrión: nuevos aspectos de una vieja enfermedad», *Acta Médica Peruana*, vol. 26, n.º 1, 2009.
[2] Ministerio de Salud del Perú, «Vigilancia epidemiológica de la enfermedad de Carrión, 2000–2024», Plataforma Nacional de Datos Abiertos. https://www.datosabiertos.gob.pe
[3] Y. LeCun, Y. Bengio y G. Hinton, «Deep learning», *Nature*, vol. 521, pp. 436–444, 2015.
[4] I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*. MIT Press, 2016.
[5] L. Breiman, J. Friedman, R. Olshen y C. Stone, *Classification and Regression Trees*. Wadsworth, 1984.
[6] J. R. Quinlan, «Induction of decision trees», *Machine Learning*, vol. 1, n.º 1, pp. 81–106, 1986.
[7] F. Pedregosa et al., «Scikit-learn: Machine learning in Python», *JMLR*, vol. 12, pp. 2825–2830, 2011.
[8] M. Abadi et al., «TensorFlow: Large-scale machine learning on heterogeneous systems», 2015. https://www.tensorflow.org
[9] F. Chollet et al., «Keras», 2015. https://keras.io
[10] D. P. Kingma y J. Ba, «Adam: A method for stochastic optimization», en *ICLR*, 2015.
[11] L. Grinsztajn, E. Oyallon y G. Varoquaux, «Why do tree-based models still outperform deep learning on typical tabular data?», en *NeurIPS*, 2022.
[12] R. Shwartz-Ziv y A. Armon, «Tabular data: Deep learning is not all you need», *Information Fusion*, vol. 81, pp. 84–90, 2022.
[13] T. Hastie, R. Tibshirani y J. Friedman, *The Elements of Statistical Learning*, 2.ª ed. Springer, 2009.
[14] V. Nair y G. E. Hinton, «Rectified linear units improve restricted Boltzmann machines», en *ICML*, 2010.
[15] N. Srivastava et al., «Dropout: A simple way to prevent neural networks from overfitting», *JMLR*, vol. 15, pp. 1929–1958, 2014.
[16] D. Smilkov et al., «TensorFlow.js: Machine learning for the web and beyond», en *MLSys*, 2019.
[17] Organización Panamericana de la Salud, «Bartonelosis (enfermedad de Carrión)» y metodología del canal endémico. https://www.paho.org
[18] V. Jiménez-Vásquez et al., «In-silico identification of linear B-cell epitopes in specific proteins of *Bartonella bacilliformis* for the serological diagnosis of Carrion's disease», *PLOS Neglected Tropical Diseases*, vol. 17, n.º 5, e0011321, 2023.
[19] G. M. Vadmal et al., «Data-driven predictions of potential Leishmania vectors in the Americas», *PLOS Neglected Tropical Diseases*, vol. 17, n.º 2, e0010749, 2023.
[20] B. Nayak et al., «Artificial intelligence (AI): a new window to revamp the vector-borne disease control», *Parasitology Research*, vol. 122, n.º 2, pp. 369–379, 2023.
[21] K. S. Rufasto-Goche et al., «Epidemiological dynamics of dengue in Peru: Temporal and spatial drivers between 2000 and 2022», *PLOS One*, vol. 20, n.º 3, e0319708, 2025.
[22] NASA Langley Research Center, «POWER: Prediction Of Worldwide Energy Resources», API de datos meteorológicos. https://power.larc.nasa.gov
[23] Instituto Nacional de Estadística e Informática (INEI), «Censos Nacionales 2017: XII de Población y VII de Vivienda», Lima, Perú.
