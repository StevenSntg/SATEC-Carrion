import os
from satec.models.figure_architecture import plot_architecture


def test_plot_architecture_crea_png(tmp_path):
    out = tmp_path / "fig_arquitectura.png"
    plot_architecture(str(out))
    assert out.exists() and out.stat().st_size > 0
