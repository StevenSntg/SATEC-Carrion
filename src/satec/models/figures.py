"""Figuras de resultados del modelado."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metrics_bar(res_df, out_path):
    metricas = ["recall", "pr_auc", "f1"]
    ax = res_df.set_index("modelo")[metricas].plot.bar(figsize=(10, 5))
    ax.set_ylabel("valor")
    ax.set_title("Metricas por modelo (conjunto de prueba)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_generalization_gap(res_df, out_path):
    sub = res_df.dropna(subset=["gap"])
    ax = sub.set_index("modelo")["gap"].plot.bar(figsize=(10, 5),
                                                 color="indianred")
    ax.set_ylabel("brecha (train_acc - test_acc)")
    ax.set_title("Brecha de generalizacion por modelo")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(repo: str = ".") -> None:
    import pandas as pd
    res = pd.read_csv(os.path.join(repo, "results", "metricas_modelos.csv"))
    plot_metrics_bar(res, os.path.join(repo, "results", "fig_metricas.png"))
    plot_generalization_gap(res, os.path.join(repo, "results", "fig_brecha.png"))
    print("[OK] figuras en results/")


if __name__ == "__main__":
    main()
