import numpy as np
import pandas as pd
from satec.models.interpret import permutation_importance_ap, calibration_points


def test_permutation_detecta_feature_relevante():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"rel": rng.rand(200), "irrel": rng.rand(200)})
    y = (X["rel"] > 0.5).astype(int)

    def pp(Xin):
        return Xin["rel"].to_numpy()  # predictor que solo usa 'rel'

    imp = permutation_importance_ap(pp, X, y, n_repeats=3)
    assert imp.iloc[0]["feature"] == "rel"  # la más importante


def test_calibration_points_en_rango():
    y = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    s = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.95]
    pp, pt = calibration_points(y, s, n_bins=3)
    assert len(pp) == len(pt)
    assert (pt >= 0).all() and (pt <= 1).all()
