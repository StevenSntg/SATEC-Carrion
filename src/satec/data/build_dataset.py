"""Orquestador del pipeline de datos núcleo de SATEC."""
import pandas as pd

from satec.config import DEFAULT_PARAMS
from satec.data.load import load_raw, clean
from satec.data.aggregate import aggregate_weekly
from satec.data.grid import build_full_grid, fill_zeros
from satec.data.endemic import select_endemic_provinces
from satec.data.endemic_channel import compute_channel
from satec.data.target import label_outbreak
from satec.features.build import add_features


def build_dataset(raw_path: str, params: dict = DEFAULT_PARAMS) -> pd.DataFrame:
    df = clean(load_raw(raw_path))
    agg = aggregate_weekly(df)

    endemic = select_endemic_provinces(
        agg, params["endemic_min_casos"], params["endemic_min_anios"])
    agg = agg.merge(endemic, on=["departamento", "provincia"], how="inner")

    panel = fill_zeros(agg, build_full_grid(agg))
    panel = compute_channel(panel, params["n_ref"], params["min_ref"])
    panel = label_outbreak(panel, params["horizon"], params["min_cases"])
    panel = add_features(panel)

    return panel.dropna(subset=["brote"]).reset_index(drop=True)


def main(raw_path: str, out_path: str) -> None:
    df = build_dataset(raw_path)
    df.to_parquet(out_path, index=False)
    print(f"[OK] dataset: {df.shape} -> {out_path}")
    print(f"  provincias endemicas: {df['provincia'].nunique()}")
    print(f"  tasa de brotes: {df['brote'].mean():.3f}")


if __name__ == "__main__":
    main("data/raw/carrion_minsa_2000_2024.csv",
         "data/processed/dataset.parquet")
