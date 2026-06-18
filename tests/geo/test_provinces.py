from satec.geo.provinces import province_centroid, provinces_table


def _feature():
    return {
        "type": "Feature",
        "properties": {"NOMBPROV": "HUARAZ", "FIRST_IDPR": "0201"},
        "geometry": {"type": "Polygon",
                     "coordinates": [[[-77.0, -9.0], [-78.0, -9.0],
                                      [-78.0, -10.0], [-77.0, -10.0],
                                      [-77.0, -9.0]]]},
    }


def test_centroid_promedia_vertices():
    lon, lat = province_centroid(_feature())
    assert abs(lon - (-77.5)) < 1e-6
    assert abs(lat - (-9.5)) < 1e-6


def test_provinces_table_columnas():
    df = provinces_table({"features": [_feature()]})
    assert list(df.columns) == ["ubigeo_prov", "nombre_prov", "lon", "lat"]
    assert df["ubigeo_prov"].iloc[0] == "0201"
