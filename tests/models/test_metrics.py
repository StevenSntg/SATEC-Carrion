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


def test_metrics_incluye_f2_y_especificidad():
    # 2 positivos: 1 detectado (tp) y 1 perdido (fn); 1 falsa alarma (fp)
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    m = evaluate_predictions(y_true, y_pred)
    assert "f2" in m and "especificidad" in m
    # especificidad = tn/(tn+fp) = 1/2
    assert abs(m["especificidad"] - 0.5) < 1e-9
    # F2 pondera más el recall que F1; con recall=0.5 y precision=0.5, F2=0.5
    assert abs(m["f2"] - 0.5) < 1e-9
