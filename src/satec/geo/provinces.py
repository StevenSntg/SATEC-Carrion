"""Carga de límites provinciales (GeoJSON) y cálculo de centroides."""
import json
import pandas as pd


def load_provinces_geojson(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _exterior_rings(geometry: dict):
    t = geometry["type"]
    coords = geometry["coordinates"]
    if t == "Polygon":
        return [coords[0]]
    if t == "MultiPolygon":
        return [poly[0] for poly in coords]
    raise ValueError(f"Geometria no soportada: {t}")


def province_centroid(feature: dict) -> tuple:
    pts = []
    for ring in _exterior_rings(feature["geometry"]):
        pts.extend(ring[:-1])  # excluye el vertice de cierre (igual al primero)
    lon = sum(p[0] for p in pts) / len(pts)
    lat = sum(p[1] for p in pts) / len(pts)
    return (lon, lat)


def provinces_table(geojson: dict) -> pd.DataFrame:
    rows = []
    for feat in geojson["features"]:
        props = feat["properties"]
        lon, lat = province_centroid(feat)
        rows.append({
            "ubigeo_prov": str(props["FIRST_IDPR"]).zfill(4),
            "nombre_prov": str(props["NOMBPROV"]).strip().upper(),
            "lon": lon, "lat": lat,
        })
    return pd.DataFrame(rows)
