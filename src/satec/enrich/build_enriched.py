"""Orquestador del enriquecimiento: clima + poblacion sobre el dataset base."""
import json
import os
import pandas as pd

from satec.enrich.climate import parse_power_json, to_weekly
from satec.enrich.climate_features import add_climate_features
from satec.enrich.population import attach_rate


def attach_ubigeo(dataset: pd.DataFrame, ubigeo_map: pd.DataFrame) -> pd.DataFrame:
    """Anade ubigeo_prov al dataset uniendo por (departamento, provincia)."""
    return dataset.merge(ubigeo_map, on=["departamento", "provincia"], how="left")


def weekly_climate_for_provinces(prov_df: pd.DataFrame,
                                 cache_dir: str) -> pd.DataFrame:
    """Parsea el cache de clima por provincia -> panel semanal con ubigeo_prov."""
    partes = []
    for row in prov_df.itertuples():
        cache_file = os.path.join(cache_dir, f"{row.lon}_{row.lat}.json")
        if not os.path.exists(cache_file):
            continue
        with open(cache_file, encoding="utf-8") as f:
            payload = json.load(f)
        weekly = to_weekly(parse_power_json(payload))
        weekly["ubigeo_prov"] = row.ubigeo_prov
        partes.append(weekly)
    return pd.concat(partes, ignore_index=True)


def build_enriched(dataset, ubigeo_map, prov_df, cache_dir,
                   pob_df=None) -> pd.DataFrame:
    df = attach_ubigeo(dataset, ubigeo_map)            # anade ubigeo_prov
    clima = weekly_climate_for_provinces(prov_df, cache_dir)
    df = df.merge(clima, on=["ubigeo_prov", "anio", "semana"], how="left")
    df = add_climate_features(df)
    if pob_df is not None:
        df = attach_rate(df, pob_df)
    return df


def build_province_table(repo: str):
    """Devuelve (dataset, ubigeo_map, prov) — provincias del dataset con centroide."""
    from satec.data.load import load_raw
    from satec.data.ubigeo import province_ubigeo_map
    from satec.geo.provinces import load_provinces_geojson, provinces_table

    dataset = pd.read_parquet(os.path.join(repo, "data/processed/dataset.parquet"))
    raw = load_raw(os.path.join(repo, "data/raw/carrion_minsa_2000_2024.csv"))
    umap = province_ubigeo_map(raw)
    geo = provinces_table(load_provinces_geojson(
        os.path.join(repo, "data/raw/peru_provincial.geojson")))

    prov = (dataset[["departamento", "provincia"]].drop_duplicates()
            .merge(umap, on=["departamento", "provincia"], how="left")
            .merge(geo[["ubigeo_prov", "lon", "lat"]],
                   on="ubigeo_prov", how="left"))
    return dataset, umap, prov


def main(repo: str = ".") -> None:
    from satec.enrich.climate_download import download_all

    dataset, umap, prov = build_province_table(repo)
    con_geo = prov.dropna(subset=["lon", "lat"]).copy()
    print(f"provincias en dataset: {len(prov)} | con centroide: {len(con_geo)}")

    cache_dir = os.path.join(repo, "data/raw/climate")
    download_all(con_geo, cache_dir=cache_dir)

    enriched = build_enriched(dataset, umap, con_geo, cache_dir)
    out = os.path.join(repo, "data/processed/dataset_enriched.parquet")
    enriched.to_parquet(out, index=False)
    print(f"[OK] enriquecido: {enriched.shape} -> {out}")
    print(f"  filas con clima (prec no nula): {enriched['prec'].notna().mean():.3f}")


if __name__ == "__main__":
    main()
