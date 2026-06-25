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


def nice_model(name):
    return MODELOS.get(name, name)


def nice_var(name):
    return VARIABLES.get(name, name)
