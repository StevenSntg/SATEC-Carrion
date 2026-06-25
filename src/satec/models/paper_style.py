"""Estilo tipografico comun para las figuras del articulo (calidad de revista).

Centraliza paleta accesible (Okabe-Ito, segura para daltonismo e impresion en
escala de grises, como exige la guia de figuras de ACM), tipografia serif acorde
al cuerpo del texto, nombres legibles de modelos y variables, y un guardado a
300 dpi. Las figuras NO llevan titulo embebido: el titulo va en el pie (caption)
del documento, como corresponde a un articulo cientifico.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta Okabe-Ito (colorblind-safe). Orden pensado para series de metricas.
AZUL = "#0072B2"
NARANJA = "#E69F00"
VERDE = "#009E73"
BERMELLON = "#D55E00"
CELESTE = "#56B4E9"
ROSA = "#CC79A7"
AMARILLO = "#F0E442"
GRIS = "#5A5A5A"
PALETA = [AZUL, NARANJA, VERDE, BERMELLON, CELESTE, ROSA]

# Nombres legibles de los modelos (con acentos), para ejes y leyendas.
MODELOS = {
    "red_neuronal": "Red neuronal",
    "arbol_poda8": "Árbol (poda 8)",
    "arbol_sin_poda": "Árbol (sin poda)",
    "baseline_persistencia": "Canal endémico",
}

# Nombres legibles de las variables del modelo, para la figura de importancia.
VARIABLES = {
    "roll_mean8": "Casos · media móvil 8 sem.",
    "roll_mean4": "Casos · media móvil 4 sem.",
    "casos": "Casos · semana actual",
    "casos_lag1": "Casos · t−1",
    "casos_lag2": "Casos · t−2",
    "casos_lag4": "Casos · t−4",
    "tasa": "Tasa por 100 000 hab.",
    "temp": "Temperatura",
    "temp_roll4": "Temperatura · MM 4 sem.",
    "temp_roll8": "Temperatura · MM 8 sem.",
    "temp_lag4": "Temperatura · t−4",
    "temp_lag8": "Temperatura · t−8",
    "hum": "Humedad relativa",
    "hum_roll4": "Humedad · MM 4 sem.",
    "hum_roll8": "Humedad · MM 8 sem.",
    "hum_lag4": "Humedad · t−4",
    "hum_lag8": "Humedad · t−8",
    "prec": "Precipitación",
    "prec_roll4": "Precipitación · MM 4 sem.",
    "prec_roll8": "Precipitación · MM 8 sem.",
    "prec_lag4": "Precipitación · t−4",
    "prec_lag8": "Precipitación · t−8",
    "sin_semana": "Estacionalidad · sen",
    "cos_semana": "Estacionalidad · cos",
}


def apply_style():
    """Aplica los rcParams de revista (serif, tamanos, grid sutil, sin marco)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def clean_axes(ax, grid_axis="y"):
    """Quita los marcos superior/derecho y deja grilla sutil en un solo eje."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x" if grid_axis == "x" else "y", which="major")
    ax.grid(axis="y" if grid_axis == "x" else "x", visible=False)


# --- Versiones en inglés (para el paper en inglés) ---
MODELOS_EN = {
    "red_neuronal": "Neural Network",
    "arbol_poda8": "Tree (depth 8)",
    "arbol_sin_poda": "Tree (unpruned)",
    "baseline_persistencia": "Endemic channel",
}

VARIABLES_EN = {
    "roll_mean8": "Cases · 8-week MA",
    "roll_mean4": "Cases · 4-week MA",
    "casos": "Cases · current week",
    "casos_lag1": "Cases · t−1",
    "casos_lag2": "Cases · t−2",
    "casos_lag4": "Cases · t−4",
    "tasa": "Rate per 100,000 inhab.",
    "temp": "Temperature",
    "temp_roll4": "Temperature · 4-week MA",
    "temp_roll8": "Temperature · 8-week MA",
    "temp_lag4": "Temperature · t−4",
    "temp_lag8": "Temperature · t−8",
    "hum": "Relative humidity",
    "hum_roll4": "Humidity · 4-week MA",
    "hum_roll8": "Humidity · 8-week MA",
    "hum_lag4": "Humidity · t−4",
    "hum_lag8": "Humidity · t−8",
    "prec": "Precipitation",
    "prec_roll4": "Precipitation · 4-week MA",
    "prec_roll8": "Precipitation · 8-week MA",
    "prec_lag4": "Precipitation · t−4",
    "prec_lag8": "Precipitation · t−8",
    "sin_semana": "Seasonality · sin",
    "cos_semana": "Seasonality · cos",
}

# Cadenas comunes de las figuras, por idioma.
TXT = {
    "es": {
        "metric_value": "Valor de la métrica", "no_outbreak": "No brote",
        "outbreak": "Brote", "prediction": "Predicción", "observed": "Observado",
        "pred_prob": "Probabilidad predicha",
        "obs_freq": "Frecuencia observada de brote",
        "perfect_calib": "Calibración perfecta",
        "aucpr_drop": "Caída en AUC-PR al permutar la variable",
        "precision": "Precisión", "recall": "Sensibilidad", "f1": "F1",
        "support": "Soporte", "accuracy": "Exactitud",
        "macro_avg": "Promedio macro", "weighted_avg": "Promedio ponderado",
        "test": "prueba", "threshold": "umbral",
        "input": "Entrada", "hidden": "Oculta", "output": "Salida",
        "vars24": "24 variables", "sigmoid": "1 · sigmoide",
        "nn_note": ("Entrada estandarizada (z-score) · pérdida de entropía cruzada "
                    "binaria · optimizador Adam · ponderación de clases"),
        "TN": "VN", "FP": "FP", "FN": "FN", "TP": "VP",
    },
    "en": {
        "metric_value": "Metric value", "no_outbreak": "No outbreak",
        "outbreak": "Outbreak", "prediction": "Prediction", "observed": "Observed",
        "pred_prob": "Predicted probability",
        "obs_freq": "Observed outbreak frequency",
        "perfect_calib": "Perfect calibration",
        "aucpr_drop": "AUC-PR drop when permuting the variable",
        "precision": "Precision", "recall": "Recall", "f1": "F1",
        "support": "Support", "accuracy": "Accuracy",
        "macro_avg": "Macro avg", "weighted_avg": "Weighted avg",
        "test": "test", "threshold": "threshold",
        "input": "Input", "hidden": "Hidden", "output": "Output",
        "vars24": "24 variables", "sigmoid": "1 · sigmoid",
        "nn_note": ("Standardized input (z-score) · binary cross-entropy loss · "
                    "Adam optimizer · class weighting"),
        "TN": "TN", "FP": "FP", "FN": "FN", "TP": "TP",
    },
}


def t(key, lang="es"):
    return TXT.get(lang, TXT["es"]).get(key, key)


def nice_model(name, lang="es"):
    d = MODELOS_EN if lang == "en" else MODELOS
    return d.get(name, name)


def nice_var(name, lang="es"):
    d = VARIABLES_EN if lang == "en" else VARIABLES
    return d.get(name, name)
