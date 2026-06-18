"""Entrena RN y AD sobre todo el dataset y los exporta para la web."""
import json
import os
import shutil
import numpy as np
import pandas as pd

from satec.models.features_matrix import feature_matrix, FEATURE_COLS
from satec.models.train import train_decision_tree, train_neural_net
from satec.web_export.export_tree import tree_to_dict

CLASS_LABELS = ["no_brote", "brote"]


def export_all(repo: str = ".") -> None:
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    X, y = feature_matrix(df)

    out_dir = os.path.join(repo, "web", "public", "models")
    os.makedirs(out_dir, exist_ok=True)

    # Arbol de Decision (poda 8) -> JSON
    ad = train_decision_tree(X, y, max_depth=8)
    with open(os.path.join(out_dir, "ad.json"), "w", encoding="utf-8") as f:
        json.dump(tree_to_dict(ad, FEATURE_COLS, CLASS_LABELS), f)

    # Red Neuronal -> TensorFlow.js + norm.json
    model, norm = train_neural_net(X, y, epochs=60)
    with open(os.path.join(out_dir, "norm.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_names": FEATURE_COLS,
                   "min": norm["min"].tolist(),
                   "rng": norm["rng"].tolist()}, f)
    rn_dir = os.path.join(out_dir, "rn")
    if os.path.exists(rn_dir):
        shutil.rmtree(rn_dir)
    os.makedirs(rn_dir, exist_ok=True)
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, rn_dir)
    print(f"[OK] modelos exportados a {out_dir}")


if __name__ == "__main__":
    export_all()
