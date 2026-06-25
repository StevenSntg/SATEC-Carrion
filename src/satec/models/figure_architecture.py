"""Figura de la arquitectura de la red neuronal (24-32-32-1) como grafo de nodos.

Diagrama de tipo capa-de-nodos (entrada / ocultas / salida) con conexiones
densas, en la paleta accesible del artículo. Los nodos son ilustrativos: el
número real de neuronas se indica en la etiqueta de cada capa."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from satec.models.paper_style import apply_style, AZUL, CELESTE, NARANJA


def plot_architecture(out_path):
    apply_style()
    # (nodos dibujados, título de capa, subtítulo, color)
    capas = [
        (6, "Entrada", "24 variables", AZUL),
        (7, "Oculta 1", "32 · ReLU", CELESTE),
        (7, "Oculta 2", "32 · ReLU", CELESTE),
        (1, "Salida", "1 · sigmoide", NARANJA),
    ]
    GAP_X = 4.0     # separación horizontal entre capas
    TOP = 6.0       # alto del área de nodos
    R = 0.32        # radio de los nodos

    fig, ax = plt.subplots(figsize=(7.8, 4.5))

    # posiciones de los nodos por capa (de abajo hacia arriba)
    pos = []
    for i, (n, *_r) in enumerate(capas):
        x = i * GAP_X
        ys = [TOP / 2] if n == 1 else list(np.linspace(0.5, TOP - 0.5, n))
        pos.append([(x, y) for y in ys])

    # conexiones densas (detrás), con el color de la capa de origen
    for i in range(len(capas) - 1):
        col = capas[i][3]
        for (x1, y1) in pos[i]:
            for (x2, y2) in pos[i + 1]:
                ax.plot([x1 + R, x2 - R], [y1, y2], color=col, alpha=0.18,
                        linewidth=0.6, zorder=1, solid_capstyle="round")

    # nodos y etiquetas de capa
    for i, (n, titulo, sub, col) in enumerate(capas):
        x = i * GAP_X
        for (xx, yy) in pos[i]:
            ax.add_patch(Circle((xx, yy), R, facecolor=col, edgecolor="white",
                                 linewidth=1.5, zorder=3))
        ax.text(x, TOP + 0.7, titulo, ha="center", va="bottom",
                fontsize=11.5, fontweight="bold")
        ax.text(x, -0.7, sub, ha="center", va="top", fontsize=10, color="#2A2A2A")

    # nota de pie con los detalles de entrenamiento (no saturan el grafo)
    cx = (len(capas) - 1) * GAP_X / 2
    ax.text(cx, -1.9,
            "Entrada estandarizada (z-score) · pérdida de entropía cruzada binaria · "
            "optimizador Adam · ponderación de clases",
            ha="center", va="top", fontsize=8.5, color="#555555", style="italic")

    ax.set_xlim(-R - 0.7, (len(capas) - 1) * GAP_X + R + 0.7)
    ax.set_ylim(-2.6, TOP + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(repo: str = ".") -> None:
    import os
    plot_architecture(os.path.join(repo, "results", "fig_arquitectura.png"))
    print("[OK] fig_arquitectura.png")


if __name__ == "__main__":
    main()
