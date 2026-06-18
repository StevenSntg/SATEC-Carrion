from satec.models.metrics import evaluate_predictions


def test_metrics_caso_perfecto():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.8]
    m = evaluate_predictions(y_true, y_pred, y_score)
    assert abs(m["recall"] - 1.0) < 1e-9
    assert abs(m["precision"] - 1.0) < 1e-9
    assert abs(m["pr_auc"] - 1.0) < 1e-9
    assert m["tp"] == 2 and m["fp"] == 0
