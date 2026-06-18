# Alerta temprana de brotes de la enfermedad de Carrión en el Perú: redes neuronales frente a árboles de decisión sobre datos de vigilancia y clima

*Título corto (encabezado): Redes neuronales y árboles de decisión para la alerta de brotes de Carrión*

**Jaqueline Alvarez Rocca**, Escuela Profesional de Ingeniería de Sistemas, Universidad Nacional Tecnológica de Lima Sur (UNTELS), Lima, Perú.
**Carlos Meza Pelaez**, Escuela Profesional de Ingeniería de Sistemas, UNTELS, Lima, Perú.
**Carlos Steven Santiago Flores**, Escuela Profesional de Ingeniería de Sistemas, UNTELS, Lima, Perú.

---

## Abstract

Carrión's disease (human bartonellosis), caused by *Bartonella bacilliformis* and transmitted by the *Lutzomyia* sand fly, is a neglected disease endemic to the Peruvian inter-Andean valleys, with a potentially lethal acute phase. We present **SATEC**, an early-warning system that predicts, at the province level and four weeks ahead, whether a zone will enter an **outbreak state** as defined by the **endemic channel** (the standard surveillance tool of PAHO/MINSA). The system is built on 25 years of national open surveillance data from the Peruvian Ministry of Health (MINSA, 2000–2024; ~46,120 case records), aggregated into a province by epidemiological-week panel with imputed zero weeks, and enriched with **climate** variables (NASA POWER: precipitation, temperature, humidity, with lags) and **population** (2017 census, for incidence rates). We compare a feed-forward **Neural Network** (Keras) against a **Decision Tree** (scikit-learn) under **strict temporal validation** (train ≤2019, test 2020–2024), reporting metrics suited to rare events (recall, AUC-PR), confusion matrices, calibration and permutation importance. The neural network attains the best balance (recall 0.81, AUC-PR 0.745, Brier 0.012) and outperforms both the unpruned tree, which overfits, and the classical endemic-channel baseline; the recent moving average of cases dominates, while climate and population add real signal. The system is deployed as an open, reproducible web application with a provincial risk map over a basemap of Peru. We conclude that machine learning can add value over classical surveillance for a neglected disease, provided the variables are informative and the evaluation respects temporal causality.

**CCS Concepts:** • Computing methodologies → Machine learning; Neural networks; Classification and regression trees. • Applied computing → Health informatics; Life and medical sciences.

**Keywords:** enfermedad de Carrión; bartonelosis; alerta temprana; redes neuronales; árboles de decisión; canal endémico; vigilancia epidemiológica; aprendizaje automático; Perú.

## Resumen

La enfermedad de Carrión (bartonelosis humana), causada por *Bartonella bacilliformis* y transmitida por el vector *Lutzomyia*, es una enfermedad desatendida y endémica de los valles interandinos del Perú, con una fase aguda potencialmente letal. Presentamos **SATEC**, un sistema de alerta temprana que predice, a nivel de **provincia** y con **cuatro semanas de anticipación**, si una zona entrará en **estado de brote** según el **canal endémico** (la herramienta estándar de vigilancia de la OPS/MINSA). El sistema se construye sobre 25 años de datos abiertos de vigilancia del Ministerio de Salud (MINSA, 2000–2024; ~46.120 registros de casos), agregados en un panel provincia por semana epidemiológica con imputación de semanas en cero, y enriquecidos con variables **climáticas** (NASA POWER: precipitación, temperatura, humedad, con rezagos) y de **población** (censo 2017, para tasas de incidencia). Se comparan una **Red Neuronal** prealimentada (Keras) y un **Árbol de Decisión** (scikit-learn) bajo **validación temporal estricta** (entrenamiento ≤2019, prueba 2020–2024), reportando métricas adecuadas a eventos raros (sensibilidad, AUC-PR), matrices de confusión, calibración e importancia por permutación. La red neuronal logra el mejor equilibrio (sensibilidad 0,81; AUC-PR 0,745; Brier 0,012) y supera tanto al árbol sin poda, que sobreajusta, como al baseline clásico del canal endémico; la media móvil reciente de casos domina, mientras que el clima y la población aportan señal real. El sistema se despliega como una aplicación web abierta y reproducible con un mapa de riesgo provincial sobre un mapa base del Perú. Concluimos que el aprendizaje automático puede aportar valor sobre la vigilancia clásica en una enfermedad desatendida, siempre que las variables sean informativas y la evaluación respete la causalidad temporal.

---

## 1. Introducción

La enfermedad de Carrión, o bartonelosis humana, es una enfermedad bacteriana desatendida y endémica de los valles interandinos del Perú, causada por *Bartonella bacilliformis* y transmitida por flebótomos del género *Lutzomyia*, principalmente *Lutzomyia verrucarum* y *Lutzomyia peruensis* [1], [17]. Lleva el nombre de Daniel Alcides Carrión, mártir de la medicina peruana. La enfermedad presenta dos fases de relevancia clínica muy distinta: una fase aguda (anemia hemolítica severa, conocida como «Fiebre de la Oroya»), de alta letalidad en ausencia de tratamiento, y una fase eruptiva («Verruga Peruana»), más benigna. Su carácter focal y estacional, ligado a las condiciones ambientales que favorecen al vector, la convierte en una candidata natural para sistemas de **alerta temprana** que anticipen la intensificación de la transmisión y orienten la respuesta de salud pública en las zonas endémicas.

El aprendizaje automático se ha consolidado como una herramienta de apoyo a la decisión en salud pública, donde conviven dos grandes familias de modelos supervisados: las redes neuronales artificiales, capaces de aproximar funciones complejas mediante combinaciones no lineales de sus entradas [3], [4], y los árboles de decisión, que clasifican mediante una secuencia de reglas interpretables sobre los atributos [5], [6]. Ambos enfoques resuelven el mismo problema de clasificación, pero difieren en su capacidad de generalización, su interpretabilidad y su sensibilidad a la calidad de los datos. En datos tabulares, el formato típico de los registros epidemiológicos, la evidencia reciente muestra que los modelos basados en árboles siguen siendo altamente competitivos frente a las redes profundas [11], [12], lo que motiva una comparación controlada sobre datos reales de una enfermedad endémica.

La aplicación de la inteligencia artificial a la enfermedad de Carrión es todavía incipiente y se ha concentrado en dos frentes: el diagnóstico de laboratorio y el estudio del vector. En el primero, Jiménez-Vásquez et al. [18] aplican un análisis *in-silico* que combina múltiples predictores computacionales para identificar epítopos lineales de células B en proteínas de *B. bacilliformis*, con el fin de mejorar el diagnóstico serológico de la enfermedad de Carrión; en la misma línea, trabajos recientes evalúan proteínas recombinantes producidas mediante sistemas de baculovirus para aumentar la sensibilidad y especificidad de los ensayos serológicos [24]. En el segundo frente, el modelado de nicho ecológico ha permitido predecir la distribución espacial de los vectores: estudios sobre *Lutzomyia peruensis* emplean algoritmos de aprendizaje automático y de máxima entropía (máquinas de vectores de soporte, MaxEnt y algoritmos genéticos de predicción de reglas) para anticipar desplazamientos de su nicho bajo escenarios de cambio climático en el Perú [25], y enfoques análogos basados en aprendizaje automático y sistemas de información geográfica se han usado para modelar la distribución de flebótomos vectores en otras regiones [26]. De forma complementaria, la detección molecular de *B. bacilliformis* en nuevas especies de *Lutzomyia* sugiere que el rango de transmisión es más amplio de lo que indican los vectores clásicos [27].

Más allá de la enfermedad de Carrión, el aprendizaje automático se aplica con éxito a enfermedades vectoriales emparentadas y al uso de datos abiertos peruanos. Vadmal et al. [19] entrenan modelos de árboles potenciados para predecir vectores potenciales de *Leishmania* en América, identificando focos en Madre de Dios; Nayak et al. [20] revisan el papel de la inteligencia artificial en el control de las enfermedades transmitidas por vectores; y Rufasto-Goche et al. [21] modelan la dinámica espacio-temporal del dengue con veintidós años de vigilancia del MINSA, la misma fuente que se emplea en este trabajo. Sin embargo, hasta donde conocemos, **ningún estudio aborda la alerta temprana de brotes de la enfermedad de Carrión mediante aprendizaje automático**: los antecedentes se centran en el diagnóstico de laboratorio o en la distribución del vector, no en la predicción del riesgo de brote por unidad territorial y temporal. Este trabajo busca atender ese vacío.

La contribución de este artículo es cuádruple. Primero, se transforma la vigilancia de casos individuales del MINSA en un panel espacio-temporal a nivel de provincia y semana epidemiológica, con imputación de las semanas sin casos, y se define el objetivo de aprendizaje mediante el **canal endémico**, integrando una herramienta epidemiológica clásica como etiqueta supervisada. Segundo, se enriquecen los datos con variables **climáticas** (NASA POWER) y de **población** (censo 2017). Tercero, se comparan una red neuronal y un árbol de decisión bajo **validación temporal estricta**, frente a un baseline epidemiológico, con métricas adecuadas a eventos raros, matrices de confusión, calibración e interpretabilidad. Cuarto, el sistema se despliega como una **aplicación web abierta y reproducible**. La Sección 2 describe los materiales y métodos; la Sección 3 presenta los resultados; la Sección 4 los discute; la Sección 5 declara las limitaciones; y la Sección 6 resume las conclusiones.

## 2. Materiales y métodos

### 2.1 Conjuntos de datos

**Fuente primaria (MINSA).** Se emplearon los datos abiertos de vigilancia epidemiológica de la enfermedad de Carrión del Ministerio de Salud del Perú [2], correspondientes al periodo 2000–2024 (~46.120 registros de casos confirmados, con departamento, provincia, ubigeo, año, semana epidemiológica, edad, sexo y fase). Cada registro corresponde a un caso; la fuente no publica las semanas sin casos, lo que se resuelve en el preprocesamiento.

**Clima (NASA POWER).** Para el centroide geográfico de cada provincia se descargaron series diarias de precipitación, temperatura a 2 m y humedad relativa de la plataforma NASA POWER [22], agregadas a semana epidemiológica y rezagadas, en reconocimiento de la respuesta diferida del vector a las condiciones ambientales.

**Población (INEI).** Se incorporó la población provincial del censo 2017 [23] para calcular la tasa de incidencia por 100.000 habitantes, definida en la Ecuación (1):

$$ \mathrm{tasa}_{p,t} = \frac{c_{p,t}}{\mathrm{poblacion}_{p}} \times 100000 $$

donde $c_{p,t}$ es el número de casos en la provincia $p$ durante la semana $t$.

### 2.2 Construcción del panel y canal endémico

Los casos se agregaron por provincia, año y semana epidemiológica (semanas 1–52; la semana 53 se reasignó a la 52). Se generó la rejilla completa de combinaciones y se imputaron en **cero** las semanas sin notificación, creando así los ejemplos negativos. El análisis se restringió a las **provincias endémicas**, definidas como aquellas con al menos 10 casos históricos en al menos 3 años distintos; se obtuvieron **61 provincias** y un panel de **69.601** observaciones provincia-semana.

El **canal endémico** es la herramienta estándar de la OPS/MINSA para describir el comportamiento esperado de una enfermedad por semana del año [17]. Para cada provincia $p$ y semana $s$ se calcularon, a partir de los **años previos** disponibles (ventana móvil de hasta cinco años, mínimo tres), los cuartiles $Q_1$, $Q_2$ y $Q_3$ de los casos históricos. Una provincia-semana se etiquetó como **brote** según la Ecuación (2):

$$ y_{p,t} = \max_{k \in \{1,2,3,4\}} \left[\, c_{p,t+k} > Q_3^{(p)} \;\wedge\; c_{p,t+k} \ge 2 \,\right] $$

es decir, si en alguna de las **cuatro semanas siguientes** los casos superan el tercer cuartil del canal (zona de epidemia) y son al menos dos. El canal de referencia se construyó exclusivamente con información anterior al punto de predicción, evitando la fuga de información temporal. La clase resultante está fuertemente desbalanceada, con aproximadamente un 7,8 % de brotes.

### 2.3 Características

El vector de entrada combina 24 variables: términos autorregresivos de casos (rezagos en $t-1$, $t-2$, $t-4$ y medias móviles de 4 y 8 semanas), variables climáticas y sus rezagos/medias móviles, la tasa por 100.000 habitantes, y la **estacionalidad**, codificada de forma cíclica mediante la Ecuación (3):

$$ \mathrm{sen}\!\left(\frac{2\pi s}{52}\right), \qquad \cos\!\left(\frac{2\pi s}{52}\right) $$

Se excluyeron deliberadamente las claves (provincia, año, semana) y el propio canal ($Q_1$, $Q_2$, $Q_3$) para no contaminar el aprendizaje con la definición del objetivo.

### 2.4 Árbol de Decisión

El árbol de decisión (scikit-learn [7]) divide recursivamente el espacio de atributos buscando, en cada nodo, la partición que maximiza la **ganancia de información**, definida a partir de la **entropía de Shannon**. Para un conjunto $S$ con proporción de clase $p_i$, la entropía se define en la Ecuación (4):

$$ H(S) = -\sum_{i} p_i \log_2 p_i $$

y la ganancia de información de un atributo $A$ que parte $S$ en subconjuntos $S_v$ en la Ecuación (5):

$$ IG(S, A) = H(S) - \sum_{v \in \mathrm{valores}(A)} \frac{|S_v|}{|S|}\, H(S_v) $$

Se evaluaron dos variantes: un árbol **sin poda** (profundidad ilimitada) y un árbol **podado** a profundidad máxima 8. El recorrido desde la raíz hasta una hoja determina la clase predicha.

### 2.5 Red Neuronal

La red neuronal prealimentada (Keras [9] sobre TensorFlow [8]) tiene una arquitectura $24 \to 32 \to 32 \to 1$. La salida de una capa $l$ se calcula según la Ecuación (6):

$$ a^{(l)} = f\!\left(W^{(l)} a^{(l-1)} + b^{(l)}\right) $$

donde $W^{(l)}$ son los pesos, $b^{(l)}$ el sesgo y $f$ la función de activación. Las capas ocultas usan la activación **ReLU** [14], Ecuación (7), y la capa de salida la **sigmoide**, Ecuación (8), que produce la probabilidad de brote:

$$ \mathrm{ReLU}(z) = \max(0, z) $$

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Antes de entrar a la red, cada variable se normaliza mediante escalado **min-max**, Ecuación (9), con parámetros estimados solo en el conjunto de entrenamiento para evitar fuga de información:

$$ x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} $$

La red se entrena minimizando la **entropía cruzada binaria**, Ecuación (10), mediante el optimizador Adam [10]:

$$ \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i) \Big] $$

### 2.6 Manejo del desbalance

Dado que los brotes son raros, ambos modelos se entrenan con ponderación de clases. El peso de la clase $c$ se define en la Ecuación (11), donde $N$ es el total de ejemplos y $N_c$ los de la clase $c$:

$$ w_c = \frac{N}{2\, N_c} $$

de modo que la clase minoritaria (brote) recibe mayor peso en la función de pérdida.

### 2.7 Validación y métricas

Se empleó **validación temporal**: entrenamiento con los años $\le 2019$ y prueba con 2020–2024, sin mezclar años entre particiones. A partir de la matriz de confusión (verdaderos y falsos positivos y negativos: $VP$, $FP$, $VN$, $FN$) se derivan las métricas de las Ecuaciones (12)–(16): precisión, sensibilidad (recall), especificidad, exactitud y puntuación $F_1$.

$$ P = \frac{VP}{VP + FP} \qquad R = \frac{VP}{VP + FN} \qquad E = \frac{VN}{VN + FP} $$

$$ A = \frac{VP + VN}{VP + VN + FP + FN} $$

$$ F_1 = 2 \cdot \frac{P \cdot R}{P + R} $$

Dado el fuerte desbalance, las métricas primarias son la sensibilidad de brotes y el **área bajo la curva de precisión-exhaustividad** (AUC-PR), aproximada por la precisión media de la Ecuación (17):

$$ \mathrm{AP} = \sum_{n} (R_n - R_{n-1})\, P_n $$

La **calibración** de las probabilidades se evaluó con el *Brier score*, Ecuación (18), y curvas de fiabilidad:

$$ BS = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 $$

La **interpretabilidad** se evaluó mediante importancia por permutación, medida como la caída en AUC-PR al permutar aleatoriamente cada variable.

### 2.8 Arquitectura del sistema y despliegue

El sistema separa un mundo de entrenamiento en Python, que se ejecuta una sola vez, de un mundo de inferencia en el navegador. La red neuronal se exporta a TensorFlow.js [16] y el árbol a un JSON plano que el navegador recorre. La aplicación web es una página estática que muestra un **mapa de riesgo coroplético** de las provincias endémicas sobre un mapa base del Perú, con el semáforo del canal endémico y un panel comparativo de los modelos; se despliega de forma estática y reproducible.

## 3. Resultados

### 3.1 Métricas comparadas

La Tabla 1 resume el desempeño sobre el conjunto de prueba temporal (2020–2024). La **red neuronal** logra el mejor equilibrio: la mayor sensibilidad (0,81), el mejor AUC-PR (0,745) y el menor Brier (0,012). El árbol con poda mejora drásticamente al árbol sin poda en AUC-PR (0,625 frente a 0,091), y ambos modelos de aprendizaje superan al baseline del canal endémico en las métricas centradas en brotes. La Figura 1 ilustra visualmente este contraste: las barras de sensibilidad, AUC-PR y F1 muestran a la red neuronal por encima del resto en AUC-PR, y al árbol sin poda como el de peor AUC-PR pese a una exactitud global alta.

**Tabla 1. Desempeño en el conjunto de prueba temporal (2020–2024).**

| Modelo | Sensibilidad | Precisión | F1 | AUC-PR | AUC-ROC | Brier |
|---|---|---|---|---|---|---|
| Red Neuronal | **0,81** | 0,26 | 0,39 | **0,745** | 0,983 | **0,012** |
| Árbol (poda 8) | 0,71 | 0,07 | 0,13 | 0,625 | 0,871 | 0,015 |
| Árbol (sin poda) | 0,60 | 0,10 | 0,17 | 0,091 | 0,792 | 0,014 |
| Canal endémico (ref.) | 0,64 | 0,29 | 0,40 | — | — | — |

![Figura 1](../results/fig_metricas.png)
**Figura 1.** Sensibilidad, AUC-PR y F1 por modelo en el conjunto de prueba. La red neuronal domina en AUC-PR, la métrica más informativa ante el fuerte desbalance de clases; el árbol sin poda exhibe el AUC-PR más bajo. Elaboración propia.

### 3.2 Matrices de confusión

La Figura 2 presenta las matrices de confusión de la red neuronal y del árbol podado sobre los 15.616 ejemplos de prueba, de los cuales 42 corresponden a brotes reales. La red neuronal recupera **34 de los 42 brotes** (verdaderos positivos), a costa de 98 falsas alarmas; el árbol podado recupera 30 brotes con 389 falsas alarmas. La diferencia en falsos positivos explica la mayor precisión de la red y su mejor F1, aun cuando ambos modelos priorizan no dejar pasar brotes (alta sensibilidad), un comportamiento deseable en un sistema de alerta donde el costo de un brote no detectado es alto.

![Figura 2](../results/fig_confusion.png)
**Figura 2.** Matrices de confusión de la red neuronal (izquierda) y el árbol podado (derecha) sobre el conjunto de prueba. Las filas son la condición observada y las columnas la predicción. Elaboración propia.

### 3.3 Métricas derivadas de la matriz de confusión

La Figura 3 desglosa, por modelo, las cuatro métricas derivadas de la Ecuación (12)–(16): precisión, sensibilidad, F1 y especificidad. Se observa el **compromiso característico** de los problemas desbalanceados: todos los modelos alcanzan una especificidad muy alta (predicen bien la ausencia de brote, la clase mayoritaria), mientras que la precisión es baja porque las pocas alertas positivas incluyen falsas alarmas. La red neuronal ofrece la combinación más equilibrada entre sensibilidad y precisión.

![Figura 3](../results/fig_f1.png)
**Figura 3.** Precisión, sensibilidad, F1 y especificidad por modelo. La especificidad es alta en todos los casos; la red neuronal equilibra mejor sensibilidad y precisión. Elaboración propia.

### 3.4 Brecha de generalización

La Figura 4 hace explícita la brecha entre la exactitud de entrenamiento y la de prueba. El **árbol sin poda** memoriza el pasado, con una exactitud de entrenamiento cercana a 0,999, y se desploma en AUC-PR sobre datos nuevos (0,091): la firma inequívoca del sobreajuste. La poda y, sobre todo, la red neuronal, generalizan de forma mucho más estable, lo que confirma sobre datos reales la advertencia metodológica de que una exactitud alta puede ser engañosa cuando las clases están desbalanceadas.

![Figura 4](../results/fig_brecha.png)
**Figura 4.** Brecha de generalización (diferencia entre exactitud de entrenamiento y de prueba) por modelo. Valores altos indican memorización; el árbol sin poda es el caso extremo. Elaboración propia.

### 3.5 Calibración

La Figura 5 presenta las curvas de fiabilidad de la red neuronal y el árbol podado. Una curva próxima a la diagonal indica que las probabilidades predichas coinciden con las frecuencias observadas. Los *Brier scores* son bajos (0,012–0,015), favorecidos por la baja prevalencia de brotes; la red neuronal presenta la mejor calibración global, aunque con margen de mejora en los deciles de alta probabilidad, donde los brotes son escasos y la estimación es más ruidosa.

![Figura 5](../results/fig_calibracion.png)
**Figura 5.** Curvas de calibración de la red neuronal y el árbol podado frente a la calibración perfecta (diagonal). Elaboración propia.

### 3.6 Importancia de variables

La Figura 6 reporta la importancia por permutación de la red neuronal. La **media móvil de ocho semanas de casos** es el predictor dominante, seguida del número de casos actuales y de la media móvil de cuatro semanas: la historia reciente de la transmisión es, como cabía esperar, el factor más informativo. De manera relevante para la hipótesis de enriquecimiento, la **tasa por 100.000 habitantes** figura entre las cinco variables más importantes, y la **temperatura** (media móvil) aporta señal apreciable. Esto indica que el clima y la estructura poblacional contribuyen a la predicción más allá de la mera autocorrelación de los casos, lo que respalda la integración multifuente para una enfermedad sensible a las condiciones ambientales del vector.

![Figura 6](../results/fig_importancia.png)
**Figura 6.** Importancia por permutación de la red neuronal (caída en AUC-PR al permutar cada variable). La media móvil de casos domina; la tasa poblacional y la temperatura aportan señal. Elaboración propia.

### 3.7 El sistema interactivo

La Figura 7 muestra la aplicación web resultante: un mapa de riesgo de las provincias endémicas sobre un mapa base del Perú, donde cada provincia se colorea según su nivel de alerta (bajo, medio o alto) para la semana más reciente. Al seleccionar una provincia, el panel lateral muestra la probabilidad de brote estimada por la red neuronal, la decisión del árbol y los casos notificados, lo que permite a un usuario de salud pública comparar de un vistazo ambos paradigmas sobre una zona concreta. El sistema es de uso libre y su código y datos son reproducibles.

![Figura 7](assets/cap_mapa_riesgo.png)
**Figura 7.** Interfaz de SATEC: mapa de riesgo provincial sobre el mapa base del Perú, con el detalle de la provincia de Ocros (Áncash) en nivel de alerta medio. Captura de la aplicación desarrollada (elaboración propia).

## 4. Discusión

Tres conclusiones emergen de los resultados. Primero, **el aprendizaje automático aporta valor sobre la vigilancia clásica**: la red neuronal supera al canal endémico en sensibilidad y AUC-PR, anticipando brotes que la regla clásica no captura, lo que sugiere que un sistema de alerta basado en datos puede complementar la vigilancia rutinaria de la enfermedad de Carrión. Segundo, **la calidad y la causalidad de las variables son decisivas**: el árbol sin poda alcanza una exactitud engañosamente alta en entrenamiento pero fracasa en datos nuevos, lo que subraya la necesidad de la validación temporal y de métricas adecuadas a eventos raros (AUC-PR, sensibilidad) en lugar de la exactitud. Tercero, **el enriquecimiento es útil**: la tasa poblacional y la temperatura aparecen entre los predictores más importantes, coherente con la biología de un vector sensible al clima.

En términos de paradigmas, la red neuronal resultó preferible para este problema desbalanceado y ruidoso, mientras que el árbol, especialmente sin poda, ilustra los riesgos del sobreajuste. La elección, sin embargo, no debería guiarse solo por el desempeño: el árbol ofrece reglas legibles, valiosas para comunicar a los tomadores de decisión por qué una provincia entra en alerta, mientras que la red opera como una caja negra cuya interpretación requiere métodos adicionales como la importancia por permutación.

## 5. Limitaciones

Conviene declarar las limitaciones con franqueza. Los datos provienen de **vigilancia pasiva**, sujeta a subnotificación y a posibles cambios en la definición de caso a lo largo de 25 años. La población se tomó del **censo 2017 como referencia constante**, sin capturar la dinámica interanual. Las variables climáticas se asignaron por **centroide provincial**, sin resolver microclimas dentro de provincias extensas. El sistema **no realiza diagnóstico clínico individual**: es una herramienta de apoyo a la vigilancia y no sustituye la confirmación de laboratorio. Finalmente, se empleó una única partición temporal; una validación de origen móvil con múltiples cortes fortalecería las conclusiones.

## 6. Conclusiones

La enfermedad de Carrión sigue siendo una amenaza desatendida para las poblaciones rurales de los valles interandinos del Perú, donde su fase aguda puede ser letal si no se trata a tiempo. Este trabajo presentó **SATEC**, el primer sistema de alerta temprana de brotes de la enfermedad de Carrión basado en aprendizaje automático, construido íntegramente sobre datos reales de vigilancia del MINSA enriquecidos con clima y población, y validado de forma temporalmente honesta. Para la enfermedad de Carrión en concreto, los hallazgos indican que: (i) es posible **anticipar con cuatro semanas la entrada de una provincia en zona de epidemia** del canal endémico con una sensibilidad del 81 %, lo que abre la puerta a focalizar la vigilancia y los recursos en las zonas y semanas de mayor riesgo; (ii) la **dinámica reciente de casos**, junto con la **temperatura** y la **densidad poblacional**, son los factores más asociados al riesgo de brote, en consonancia con la ecología del vector *Lutzomyia*; y (iii) un modelo de aprendizaje automático puede **superar al canal endémico clásico** que hoy guía la vigilancia, sin reemplazarlo, sino complementándolo con una capa predictiva. El sistema resultante es reproducible, verificable y desplegable sin servidor, lo que lo hace viable como instrumento de apoyo para las Direcciones Regionales de Salud de las zonas endémicas. Como trabajo futuro se plantea la validación de origen móvil, la incorporación de modelos de ensamble y de variables entomológicas, el descenso a granularidad distrital y la validación prospectiva en campo, así como la articulación con los esfuerzos de diagnóstico serológico y de modelado del vector que hoy concentran la investigación computacional sobre la enfermedad de Carrión.

## Agradecimientos

Los autores agradecen a la Escuela Profesional de Ingeniería de Sistemas de la Universidad Nacional Tecnológica de Lima Sur (UNTELS) por el apoyo institucional, y al Ministerio de Salud del Perú (MINSA) por la publicación de los datos abiertos de vigilancia epidemiológica que hicieron posible este estudio. Asimismo, reconocen la labor del personal de salud que sostiene la vigilancia de la enfermedad de Carrión en las zonas endémicas del país.

## Disponibilidad de datos y código

Los datos de vigilancia provienen de los datos abiertos del MINSA [2]; las variables climáticas, de NASA POWER [22]; la población, del censo del INEI [23]; y los límites provinciales, de un repositorio GeoJSON público. El código del pipeline de datos, los modelos y la aplicación web son reproducibles de extremo a extremo con Python 3.12 y se acompañan de pruebas automatizadas.

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
[18] V. Jiménez-Vásquez, K. D. Calvay-Sánchez, Y. Zárate-Sulca y G. Mendoza-Mujica, «In-silico identification of linear B-cell epitopes in specific proteins of *Bartonella bacilliformis* for the serological diagnosis of Carrion's disease», *PLOS Neglected Tropical Diseases*, vol. 17, n.º 5, e0011321, 2023.
[19] G. M. Vadmal et al., «Data-driven predictions of potential Leishmania vectors in the Americas», *PLOS Neglected Tropical Diseases*, vol. 17, n.º 2, e0010749, 2023.
[20] B. Nayak et al., «Artificial intelligence (AI): a new window to revamp the vector-borne disease control», *Parasitology Research*, vol. 122, n.º 2, pp. 369–379, 2023.
[21] K. S. Rufasto-Goche et al., «Epidemiological dynamics of dengue in Peru: Temporal and spatial drivers between 2000 and 2022», *PLOS One*, vol. 20, n.º 3, e0319708, 2025.
[22] NASA Langley Research Center, «POWER: Prediction Of Worldwide Energy Resources», API de datos meteorológicos. https://power.larc.nasa.gov
[23] Instituto Nacional de Estadística e Informática (INEI), «Censos Nacionales 2017: XII de Población y VII de Vivienda», Lima, Perú.
[24] Estudio sobre producción de proteínas de *Bartonella bacilliformis* asistida por baculovirus para mejorar el diagnóstico serológico de la enfermedad de Carrión, *PLOS Neglected Tropical Diseases*, 2024.
[25] D. Moo-Llanes et al., «Shifts in the ecological niche of *Lutzomyia peruensis* under climate change scenarios in Peru», *Medical and Veterinary Entomology*, 2017.
[26] A. A. Hanafi-Bojd et al., «Machine learning approaches in GIS-based ecological modeling of the sand fly *Phlebotomus papatasi*, a vector of zoonotic cutaneous leishmaniasis», *Acta Tropica* (ScienceDirect), 2019.
[27] J. Del Valle-Mendoza et al., «Molecular detection of *Bartonella bacilliformis* in *Lutzomyia maranonensis* in Cajamarca, Peru: a new potential vector of Carrion's disease?», 2018.
