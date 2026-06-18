from satec.enrich.climate import parse_power_json, to_weekly


def _payload():
    return {"properties": {"parameter": {
        "PRECTOTCORR": {"20100104": 1.0, "20100105": 2.0, "20100106": -999.0},
        "T2M": {"20100104": 10.0, "20100105": 12.0, "20100106": 11.0},
        "RH2M": {"20100104": 60.0, "20100105": 62.0, "20100106": 64.0},
    }}}


def test_parse_power_json_marca_fill_value_como_nan():
    df = parse_power_json(_payload())
    assert list(df.columns) == ["fecha", "prec", "temp", "hum"]
    assert df["prec"].isna().sum() == 1  # -999.0 -> NaN


def test_to_weekly_suma_prec_y_promedia_temp():
    weekly = to_weekly(parse_power_json(_payload()))
    fila = weekly.iloc[0]
    assert abs(fila["prec"] - 3.0) < 1e-9          # 1+2 (NaN ignorado)
    assert abs(fila["temp"] - 11.0) < 1e-9         # media de 10,12,11
