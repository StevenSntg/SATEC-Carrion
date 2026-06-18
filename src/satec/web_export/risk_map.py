"""Precomputa el riesgo por provincia (ultima semana) para el mapa web."""
import json
import os
import pandas as pd

from satec.models.features_matrix import feature_matrix
from satec.models.train import (train_decision_tree, train_neural_net,
                                nn_predict_proba)


def latest_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    idx = (df.sort_values(["anio", "semana"])
             .groupby("ubigeo_prov").tail(1).index)
    return df.loc[idx].reset_index(drop=True)


def _nivel(prob: float) -> str:
    if prob >= 0.66:
        return "alto"
    if prob >= 0.33:
        return "medio"
    return "bajo"


def build_risk_json(repo: str = ".") -> None:
    df = pd.read_parquet(
        os.path.join(repo, "data/processed/dataset_enriched.parquet"))
    X, y = feature_matrix(df)
    ad = train_decision_tree(X, y, max_depth=8)
    model, norm = train_neural_net(X, y, epochs=60)

    latest = latest_risk_table(df)
    Xl, _ = feature_matrix(latest)
    prob_rn = nn_predict_proba(model, Xl, norm)
    pred_ad = ad.predict(Xl)

    riesgo = {}
    for i, row in latest.iterrows():
        p = float(prob_rn[i])
        riesgo[row["ubigeo_prov"]] = {
            "provincia": str(row["provincia"]),
            "prob_rn": round(p, 3), "pred_ad": int(pred_ad[i]),
            "casos": int(row["casos"]), "nivel": _nivel(p)}

    out_dir = os.path.join(repo, "web", "public", "data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "riesgo.json"), "w", encoding="utf-8") as f:
        json.dump(riesgo, f, ensure_ascii=False)

    # GeoJSON solo de provincias endemicas presentes en el dataset
    geo_path = os.path.join(repo, "data/raw/peru_provincial.geojson")
    with open(geo_path, encoding="utf-8") as f:
        geo = json.load(f)
    endemicas = set(df["ubigeo_prov"].unique())
    geo["features"] = [ft for ft in geo["features"]
                       if str(ft["properties"]["FIRST_IDPR"]).zfill(4) in endemicas]
    with open(os.path.join(out_dir, "provincias.geojson"), "w",
              encoding="utf-8") as f:
        json.dump(geo, f, ensure_ascii=False)
    print(f"[OK] riesgo.json ({len(riesgo)} provincias) y provincias.geojson "
          f"({len(geo['features'])} poligonos)")


if __name__ == "__main__":
    build_risk_json()
