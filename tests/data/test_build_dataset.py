from satec.data.build_dataset import build_dataset


def test_build_dataset_desde_fixture(tmp_path, sample_raw):
    raw = tmp_path / "raw.csv"
    sample_raw.to_csv(raw, index=False)
    params = {"horizon": 1, "n_ref": 1, "min_ref": 1, "min_cases": 1,
              "endemic_min_casos": 1, "endemic_min_anios": 1}
    df = build_dataset(str(raw), params=params)
    assert len(df) > 0
    # columnas clave presentes y sin brote nulo
    for col in ["departamento", "provincia", "anio", "semana", "casos",
                "brote", "casos_lag1", "sin_semana"]:
        assert col in df.columns
    assert df["brote"].isna().sum() == 0
