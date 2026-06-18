import json
from satec.enrich.climate_download import power_url, fetch_point


def test_power_url_incluye_parametros():
    url = power_url(-77.5, -9.5, "20000101", "20001231")
    assert "PRECTOTCORR" in url and "T2M" in url and "RH2M" in url
    assert "longitude=-77.5" in url and "latitude=-9.5" in url


def test_fetch_point_usa_cache(tmp_path):
    # pre-creamos el archivo de cache -> no debe tocar la red
    cached = {"properties": {"parameter": {"PRECTOTCORR": {}}}}
    (tmp_path / "-77.5_-9.5.json").write_text(json.dumps(cached))
    out = fetch_point(-77.5, -9.5, cache_dir=str(tmp_path))
    assert out == cached
