"""Descarga de clima de NASA POWER por punto, con cache en disco."""
import json
import os
import time
import urllib.request

_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
_PARAMS = "PRECTOTCORR,T2M,RH2M"


def power_url(lon: float, lat: float, start: str, end: str) -> str:
    return (f"{_BASE}?parameters={_PARAMS}&community=AG"
            f"&longitude={lon}&latitude={lat}"
            f"&start={start}&end={end}&format=JSON")


def fetch_point(lon, lat, start="20000101", end="20241231",
                cache_dir="data/raw/climate") -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{lon}_{lat}.json")
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    url = power_url(lon, lat, start, end)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


def download_all(provinces_df, cache_dir="data/raw/climate") -> None:
    n = len(provinces_df)
    for i, row in enumerate(provinces_df.itertuples(), 1):
        fetch_point(row.lon, row.lat, cache_dir=cache_dir)
        print(f"  [{i}/{n}] ubigeo {row.ubigeo_prov} listo")
        time.sleep(0.5)  # cortesia con la API
