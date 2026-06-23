"""Genera los notebooks Colab/Jupyter de entrenamiento y prueba de cada modelo.

Crea tres notebooks autocontenidos (no dependen del paquete `satec`): descargan el
panel enriquecido desde GitHub y reproducen el train/test del articulo SATEC.

    py -3.12 notebooks/build_notebooks.py
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = ("https://github.com/StevenSntg/SATEC-Carrion/raw/main/"
       "data/processed/dataset_enriched.parquet")
COLAB = "https://colab.research.google.com/github/StevenSntg/SATEC-Carrion/blob/main/notebooks"

# Bloques de codigo comunes -------------------------------------------------

FEATURES_SRC = '''\
# Las 24 variables de entrada (autorregresivas de casos, estacionalidad ciclica,
# clima con rezagos/medias moviles y tasa por 100 000 hab.). Se EXCLUYEN las
# claves, el objetivo y el canal endemico (q1,q2,q3) para no filtrar la etiqueta.
FEATURE_COLS = [
    "casos", "casos_lag1", "casos_lag2", "casos_lag4",
    "roll_mean4", "roll_mean8", "sin_semana", "cos_semana",
    "prec", "temp", "hum",
    "prec_lag4", "prec_lag8", "prec_roll4", "prec_roll8",
    "temp_lag4", "temp_lag8", "temp_roll4", "temp_roll8",
    "hum_lag4", "hum_lag8", "hum_roll4", "hum_roll8",
    "tasa",
]

def feature_matrix(d):
    X = d[FEATURE_COLS].astype(float).fillna(0.0)
    y = d["brote"].astype(int)
    return X, y'''

SPLIT_SRC = '''\
# Validacion TEMPORAL estricta: se entrena con <= 2019 y se prueba con 2020-2024.
# No se mezclan anios entre particiones (evita fuga de informacion temporal).
train = df[df["anio"] <= 2019].reset_index(drop=True)
test  = df[df["anio"] >  2019].reset_index(drop=True)
Xtr, ytr = feature_matrix(train)
Xte, yte = feature_matrix(test)
print("Entrenamiento:", Xtr.shape, "| brotes:", int(ytr.sum()))
print("Prueba       :", Xte.shape, "| brotes:", int(yte.sum()),
      f"({100*yte.mean():.1f}% de la clase positiva)")'''

METRICS_SRC = '''\
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss)

def evaluar(y_true, y_pred, y_score=None):
    """Metricas adecuadas a eventos raros (la prevalencia de brotes cae del
    ~10% (<=2019) al ~0.3% en 2020-2024 por la subnotificacion pandemica)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "Exactitud":     accuracy_score(y_true, y_pred),
        "Precision":     precision_score(y_true, y_pred, zero_division=0),
        "Sensibilidad":  recall_score(y_true, y_pred, zero_division=0),
        "F1":            f1_score(y_true, y_pred, zero_division=0),
        "VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp),
    }
    if y_score is not None:
        m["AUC-ROC"] = roc_auc_score(y_true, y_score)
        m["AUC-PR"]  = average_precision_score(y_true, y_score)
        m["Brier"]   = brier_score_loss(y_true, y_score)
    return m'''

CONF_PLOT_SRC = '''\
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm / cm.max(), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], ","), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()*0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No brote", "Brote"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No brote", "Brote"])
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Observado"); ax.set_title(titulo)
    plt.tight_layout(); plt.show()'''

BASELINE_SRC = '''\
# Baseline epidemiologico clasico: el "canal endemico" de la OPS/MINSA marca
# brote si los casos de hoy superan el tercer cuartil historico (q3).
bp_pred = (test["casos"] > test["q3"]).astype(int).to_numpy()
print("Canal endemico (referencia):")
print({k: round(v, 3) if isinstance(v, float) else v
       for k, v in evaluar(yte, bp_pred).items()})'''


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def build_arbol():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"""# SATEC · Árbol de Decisión — entrenamiento y prueba

[![Open In Colab]({ 'https://colab.research.google.com/assets/colab-badge.svg' })]({COLAB}/01_arbol_decision_train_test.ipynb)

**Alerta temprana de brotes de la enfermedad de Carrión.** Este cuaderno entrena
y evalúa un **Árbol de Decisión** (scikit-learn) que predice, por provincia y con
4 semanas de anticipación, si una zona entrará en **estado de brote** según el
canal endémico. Validación **temporal estricta** (entrenamiento ≤ 2019, prueba
2020–2024) y métricas adecuadas a **eventos raros**.

Repositorio: <https://github.com/StevenSntg/SATEC-Carrion>"""),
        md("## 1. Dependencias y datos"),
        code('# En Colab pandas/sklearn/matplotlib ya estan; pyarrow lee el .parquet\n'
             '!pip -q install pyarrow 2>/dev/null\n'
             'import pandas as pd, numpy as np'),
        code(f'URL = "{RAW}"\n'
             'df = pd.read_parquet(URL)\n'
             'print("Panel provincia-semana:", df.shape)\n'
             'df[["departamento","provincia","anio","semana","casos","q3","brote"]].head()'),
        md("## 2. Variables de entrada y partición temporal"),
        code(FEATURES_SRC),
        code(SPLIT_SRC),
        md("## 3. Entrenamiento del Árbol de Decisión\n\n"
           "Se usa **entropía** (ganancia de información) y `class_weight=\"balanced\"` "
           "para compensar el desbalance. Se entrenan dos variantes: **podado** "
           "(profundidad 8) y **sin poda** (para ilustrar el sobreajuste)."),
        code('from sklearn.tree import DecisionTreeClassifier\n\n'
             'arbol = DecisionTreeClassifier(criterion="entropy", max_depth=8,\n'
             '                               class_weight="balanced", random_state=42)\n'
             'arbol.fit(Xtr, ytr)\n\n'
             'arbol_sin_poda = DecisionTreeClassifier(criterion="entropy", max_depth=None,\n'
             '                                        class_weight="balanced", random_state=42)\n'
             'arbol_sin_poda.fit(Xtr, ytr)\n'
             'print("Profundidad podado:", arbol.get_depth(),\n'
             '      "| sin poda:", arbol_sin_poda.get_depth())'),
        md("## 4. Prueba (test 2020–2024)"),
        code(METRICS_SRC),
        code('for nombre, clf in [("Árbol (poda 8)", arbol), ("Árbol (sin poda)", arbol_sin_poda)]:\n'
             '    score = clf.predict_proba(Xte)[:, 1]\n'
             '    pred  = clf.predict(Xte)\n'
             '    train_acc = (clf.predict(Xtr) == ytr.to_numpy()).mean()\n'
             '    m = evaluar(yte, pred, score)\n'
             '    print(f"\\n=== {nombre} ===")\n'
             '    for k, v in m.items():\n'
             '        print(f"  {k:12s}: {v:.3f}" if isinstance(v, float) else f"  {k:12s}: {v}")\n'
             '    print(f"  Exactitud entren.: {train_acc:.3f}  (brecha = {train_acc - m[\'Exactitud\']:+.3f})")'),
        md("> El árbol **sin poda** memoriza el entrenamiento (exactitud ≈ 0,999) y se "
           "desploma en AUC-PR sobre datos nuevos: la firma del **sobreajuste**. El "
           "árbol **podado** generaliza mucho mejor."),
        md("## 5. Matriz de confusión e importancia de variables"),
        code(CONF_PLOT_SRC),
        code('plot_confusion(yte, arbol.predict(Xte), "Árbol (poda 8) — prueba 2020–2024")'),
        code('# Importancia de Gini nativa del arbol (el articulo usa importancia por\n'
             '# permutacion sobre AUC-PR, mas robusta, pero esta ilustra el mismo patron).\n'
             'imp = pd.Series(arbol.feature_importances_, index=FEATURE_COLS).sort_values()\n'
             'imp.tail(12).plot.barh(figsize=(7, 4.5), color="#0072B2")\n'
             'plt.xlabel("Importancia (Gini)"); plt.title("Árbol (poda 8) — variables")\n'
             'plt.tight_layout(); plt.show()'),
        code('# Reglas del arbol (limitadas a profundidad 3 para que sean legibles)\n'
             'from sklearn.tree import export_text\n'
             'print(export_text(arbol, feature_names=list(FEATURE_COLS), max_depth=3))'),
        md("## 6. Comparación con el baseline (canal endémico)"),
        code(BASELINE_SRC),
        md("**Conclusión.** El árbol podado supera al canal endémico clásico en las "
           "métricas centradas en brotes (sensibilidad y AUC-PR), aunque genera más "
           "falsas alarmas. Véase el cuaderno de la **Red Neuronal** para el modelo "
           "con mejor equilibrio global."),
        md("## 7. Validación de origen móvil (umbral óptimo)\n\n"
           "Esquema **train ≤ Y−2 / val = Y−1 / test = Y** para cada año Y ∈ 2016–2024. "
           "El umbral se selecciona sobre la validación maximizando F1."),
        code('from sklearn.metrics import fbeta_score\n'
             'import numpy as np\n'
             'def best_t(yv, sv, beta=1.0):\n'
             '    if int(np.sum(yv))==0: return 0.5\n'
             '    cand=np.unique(np.concatenate([[0.0],np.sort(sv),[1.0]]))\n'
             '    return float(max(cand, key=lambda t: fbeta_score(yv,(sv>=t).astype(int),beta=beta,zero_division=0)))\n'
             'yt=[];yp=[];ys=[]\n'
             'for Y in range(2016,2025):\n'
             '    tr=df[df.anio<=Y-2]; va=df[df.anio==Y-1]; te=df[df.anio==Y]\n'
             '    if len(tr)==0 or len(te)==0: continue\n'
             '    Xtr2,ytr2=feature_matrix(tr); clf2=DecisionTreeClassifier(criterion="entropy",max_depth=8,class_weight="balanced",random_state=42).fit(Xtr2,ytr2)\n'
             '    Xv,yv=feature_matrix(va); t=best_t(yv.to_numpy(),clf2.predict_proba(Xv)[:,1]) if len(va) else 0.5\n'
             '    Xte2,yte2=feature_matrix(te); s=clf2.predict_proba(Xte2)[:,1]\n'
             '    yt+=list(yte2);ys+=list(s);yp+=list((s>=t).astype(int))\n'
             'print("Origen movil — Árbol:", evaluar(np.array(yt),np.array(yp),np.array(ys)))'),
    ]
    return nb


def build_red():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"""# SATEC · Red Neuronal — entrenamiento y prueba

[![Open In Colab]({ 'https://colab.research.google.com/assets/colab-badge.svg' })]({COLAB}/02_red_neuronal_train_test.ipynb)

**Alerta temprana de brotes de la enfermedad de Carrión.** Este cuaderno entrena
y evalúa una **Red Neuronal** prealimentada (Keras/TensorFlow) con arquitectura
**24 → 32 → 32 → 1**, activaciones ReLU y salida sigmoide, optimizador Adam y
entropía cruzada binaria. Validación **temporal estricta** (entrenamiento ≤ 2019,
prueba 2020–2024).

Repositorio: <https://github.com/StevenSntg/SATEC-Carrion>"""),
        md("## 1. Dependencias y datos"),
        code('!pip -q install pyarrow 2>/dev/null\n'
             'import pandas as pd, numpy as np, tensorflow as tf\n'
             'from tensorflow import keras\n'
             'from tensorflow.keras import layers\n'
             'tf.keras.utils.set_random_seed(42)   # reproducibilidad\n'
             'print("TensorFlow", tf.__version__)'),
        code(f'URL = "{RAW}"\n'
             'df = pd.read_parquet(URL)\n'
             'print("Panel provincia-semana:", df.shape)\n'
             'df[["departamento","provincia","anio","semana","casos","q3","brote"]].head()'),
        md("## 2. Variables de entrada y partición temporal"),
        code(FEATURES_SRC),
        code(SPLIT_SRC),
        md("## 3. Normalización min-max\n\n"
           "Cada variable se escala al rango [0, 1] con parámetros estimados **solo "
           "en el entrenamiento**, para evitar fuga de información hacia la prueba."),
        code('xmin = Xtr.min().to_numpy(); xmax = Xtr.max().to_numpy()\n'
             'rng = np.where((xmax - xmin) == 0, 1.0, xmax - xmin)\n'
             'Xtr_n = (Xtr.to_numpy() - xmin) / rng\n'
             'Xte_n = (Xte.to_numpy() - xmin) / rng'),
        md("## 4. Definición y entrenamiento de la red\n\n"
           "Se ponderan las clases (`class_weight`) para que la clase minoritaria "
           "(brote) pese más en la función de pérdida."),
        code('from sklearn.utils.class_weight import compute_class_weight\n'
             'w = compute_class_weight("balanced", classes=np.array([0, 1]), y=ytr)\n'
             'class_weight = {0: float(w[0]), 1: float(w[1])}\n\n'
             'model = keras.Sequential([\n'
             '    layers.Input(shape=(Xtr_n.shape[1],)),\n'
             '    layers.Dense(32, activation="relu"),\n'
             '    layers.Dense(32, activation="relu"),\n'
             '    layers.Dense(1, activation="sigmoid"),\n'
             '])\n'
             'model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])\n'
             'model.summary()'),
        code('hist = model.fit(Xtr_n, ytr.to_numpy(), epochs=60, batch_size=256,\n'
             '                 class_weight=class_weight, verbose=0)\n'
             'import matplotlib.pyplot as plt\n'
             'plt.figure(figsize=(6, 3.5))\n'
             'plt.plot(hist.history["loss"], color="#0072B2")\n'
             'plt.xlabel("Época"); plt.ylabel("Pérdida (entropía cruzada)")\n'
             'plt.title("Curva de entrenamiento"); plt.tight_layout(); plt.show()'),
        md("## 5. Prueba (test 2020–2024)"),
        code(METRICS_SRC),
        code('score = model.predict(Xte_n, verbose=0).ravel()\n'
             'pred  = (score >= 0.5).astype(int)\n'
             'm = evaluar(yte, pred, score)\n'
             'print("=== Red Neuronal — prueba 2020-2024 ===")\n'
             'for k, v in m.items():\n'
             '    print(f"  {k:12s}: {v:.3f}" if isinstance(v, float) else f"  {k:12s}: {v}")'),
        md("## 6. Matriz de confusión y calibración"),
        code(CONF_PLOT_SRC),
        code('plot_confusion(yte, pred, "Red Neuronal — prueba 2020–2024")'),
        code('# Curva de calibracion: ¿las probabilidades predichas coinciden con la\n'
             '# frecuencia observada de brotes?\n'
             'from sklearn.calibration import calibration_curve\n'
             'pt, pp = calibration_curve(yte, score, n_bins=8, strategy="quantile")\n'
             'plt.figure(figsize=(5, 5))\n'
             'plt.plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")\n'
             'plt.plot(pp, pt, "o-", color="#0072B2", label="Red Neuronal")\n'
             'plt.xlabel("Probabilidad predicha"); plt.ylabel("Frecuencia observada de brote")\n'
             'plt.legend(); plt.title("Calibración"); plt.tight_layout(); plt.show()'),
        md("## 7. Comparación con el baseline (canal endémico)"),
        code(BASELINE_SRC),
        md("**Conclusión.** La red neuronal logra el **mejor equilibrio** "
           "(sensibilidad ≈ 0,81; AUC-PR ≈ 0,745; Brier ≈ 0,012), superando al árbol "
           "y al canal endémico clásico. Los valores exactos pueden variar levemente "
           "según la versión de TensorFlow y el hardware."),
        md("## 8. Validación de origen móvil (umbral óptimo)\n\n"
           "Esquema **train ≤ Y−2 / val = Y−1 / test = Y** para cada año Y ∈ 2016–2024. "
           "El umbral se selecciona sobre la validación maximizando F1."),
        code('from sklearn.metrics import fbeta_score\n'
             'import numpy as np\n'
             'def best_t(yv, sv, beta=1.0):\n'
             '    if int(np.sum(yv))==0: return 0.5\n'
             '    cand=np.unique(np.concatenate([[0.0],np.sort(sv),[1.0]]))\n'
             '    return float(max(cand, key=lambda t: fbeta_score(yv,(sv>=t).astype(int),beta=beta,zero_division=0)))\n'
             'yt=[];yp=[];ys=[]\n'
             'for Y in range(2016,2025):\n'
             '    tr=df[df.anio<=Y-2]; va=df[df.anio==Y-1]; te=df[df.anio==Y]\n'
             '    if len(tr)==0 or len(te)==0: continue\n'
             '    Xtr2,ytr2=feature_matrix(tr)\n'
             '    mn=Xtr2.min().to_numpy(); mx=Xtr2.max().to_numpy(); rng2=np.where((mx-mn)==0,1.0,mx-mn)\n'
             '    Xtr2n=(Xtr2.to_numpy()-mn)/rng2\n'
             '    w2=compute_class_weight("balanced",classes=np.array([0,1]),y=ytr2); cw2={0:float(w2[0]),1:float(w2[1])}\n'
             '    m2=keras.Sequential([layers.Input(shape=(Xtr2n.shape[1],)),layers.Dense(32,activation="relu"),layers.Dense(32,activation="relu"),layers.Dense(1,activation="sigmoid")])\n'
             '    m2.compile(loss="binary_crossentropy",optimizer="adam"); m2.fit(Xtr2n,ytr2.to_numpy(),epochs=30,batch_size=256,class_weight=cw2,verbose=0)\n'
             '    Xv,yv=feature_matrix(va); Xvn=(Xv.to_numpy()-mn)/rng2\n'
             '    sv=m2.predict(Xvn,verbose=0).ravel(); t=best_t(yv.to_numpy(),sv) if len(va) else 0.5\n'
             '    Xte2,yte2=feature_matrix(te); Xte2n=(Xte2.to_numpy()-mn)/rng2\n'
             '    s=m2.predict(Xte2n,verbose=0).ravel()\n'
             '    yt+=list(yte2);ys+=list(s);yp+=list((s>=t).astype(int))\n'
             'print("Origen movil — Red Neuronal:", evaluar(np.array(yt),np.array(yp),np.array(ys)))'),
    ]
    return nb


def build_rf():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"""# SATEC · Random Forest & Gradient Boosting — entrenamiento y prueba

[![Open In Colab]({ 'https://colab.research.google.com/assets/colab-badge.svg' })]({COLAB}/03_random_forest_train_test.ipynb)

**Alerta temprana de brotes de la enfermedad de Carrión.** Este cuaderno entrena
y evalúa un **Random Forest** y un **Gradient Boosting Histograma** (scikit-learn)
que predicen, por provincia y con 4 semanas de anticipación, si una zona entrará
en **estado de brote**. Validación **temporal estricta** (entrenamiento ≤ 2019,
prueba 2020–2024) y métricas adecuadas a **eventos raros**.

Repositorio: <https://github.com/StevenSntg/SATEC-Carrion>"""),
        md("## 1. Dependencias y datos"),
        code('!pip -q install pyarrow 2>/dev/null\n'
             'import pandas as pd, numpy as np'),
        code(f'URL = "{RAW}"\n'
             'df = pd.read_parquet(URL)\n'
             'print("Panel provincia-semana:", df.shape)\n'
             'df[["departamento","provincia","anio","semana","casos","q3","brote"]].head()'),
        md("## 2. Variables de entrada y partición temporal"),
        code(FEATURES_SRC),
        code(SPLIT_SRC),
        md("## 3. Entrenamiento: Random Forest y Gradient Boosting\n\n"
           "Ambos modelos usan `class_weight=\"balanced\"` para compensar el fuerte "
           "desbalance de clases (la prevalencia de brotes cae del ~10 % en ≤ 2019 "
           "al ~0,3 % en 2020–2024 por la subnotificación pandémica)."),
        code('from sklearn.ensemble import (RandomForestClassifier,\n'
             '                              HistGradientBoostingClassifier)\n\n'
             'rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",\n'
             '                            random_state=42, n_jobs=-1).fit(Xtr, ytr)\n'
             'gb = HistGradientBoostingClassifier(max_iter=300,\n'
             '          class_weight="balanced", random_state=42).fit(Xtr, ytr)'),
        md("## 4. Prueba (test 2020–2024)"),
        code(METRICS_SRC),
        code('for nombre, clf in [("Random Forest", rf), ("Gradient Boosting", gb)]:\n'
             '    score = clf.predict_proba(Xte)[:, 1]\n'
             '    pred  = clf.predict(Xte)\n'
             '    m = evaluar(yte, pred, score)\n'
             '    print(f"\\n=== {nombre} ===")\n'
             '    for k, v in m.items():\n'
             '        print(f"  {k:12s}: {v:.3f}" if isinstance(v, float) else f"  {k:12s}: {v}")'),
        md("## 5. Importancia de variables y matriz de confusión"),
        code(CONF_PLOT_SRC),
        code('plot_confusion(yte, rf.predict(Xte), "Random Forest — prueba 2020–2024")'),
        code('plot_confusion(yte, gb.predict(Xte), "Gradient Boosting — prueba 2020–2024")'),
        code('# Importancia media de impureza (MDI) del Random Forest.\n'
             'import matplotlib.pyplot as plt\n'
             'imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()\n'
             'imp.tail(12).plot.barh(figsize=(7, 4.5), color="#009E73")\n'
             'plt.xlabel("Importancia (MDI)"); plt.title("Random Forest — variables")\n'
             'plt.tight_layout(); plt.show()'),
        md("## 6. Comparación con el baseline (canal endémico)"),
        code(BASELINE_SRC),
        md("## 7. Validación de origen móvil (umbral óptimo)\n\n"
           "Esquema **train ≤ Y−2 / val = Y−1 / test = Y** para cada año Y ∈ 2016–2024. "
           "El umbral se selecciona sobre la validación maximizando F1."),
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
    ]
    return nb


def main():
    for nb, name in [(build_arbol(), "01_arbol_decision_train_test.ipynb"),
                     (build_red(),   "02_red_neuronal_train_test.ipynb"),
                     (build_rf(),    "03_random_forest_train_test.ipynb")]:
        nb.metadata = {
            "language_info": {"name": "python"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "colab": {"provenance": []},
        }
        path = os.path.join(HERE, name)
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("[OK]", path)


if __name__ == "__main__":
    main()
