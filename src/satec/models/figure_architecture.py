"""Figura esquemática de la arquitectura de la red neuronal (24-32-32-1)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from satec.models.paper_style import apply_style, AZUL, NARANJA, VERDE, GRIS


def _caja(ax, x, y, w, h, texto, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.2, edgecolor=color,
                                facecolor=color + "22"))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=10)


def plot_architecture(out_path):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    capas = [("Entrada\n24 variables\n(normalización\nmin-max)", AZUL),
             ("Oculta 1\n32 · ReLU", NARANJA),
             ("Oculta 2\n32 · ReLU", NARANJA),
             ("Salida\n1 · sigmoide\nP(brote)", VERDE)]
    x = 0.3
    centros = []
    for texto, color in capas:
        _caja(ax, x, 0.9, 1.5, 1.4, texto, color)
        centros.append(x + 1.5)
        x += 2.3
    for cx in centros[:-1]:
        ax.add_patch(FancyArrowPatch((cx, 1.6), (cx + 0.8, 1.6),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=GRIS, linewidth=1.2))
    ax.text(x + 0.1, 1.6,
            "Pérdida:\nentropía cruzada\nbinaria · Adam\nclass_weight\nbalanceado",
            ha="left", va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, x + 1.8); ax.set_ylim(0.4, 2.8)
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
