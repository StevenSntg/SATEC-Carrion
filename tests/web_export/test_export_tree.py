from sklearn.tree import DecisionTreeClassifier
import numpy as np
from satec.web_export.export_tree import tree_to_dict


def test_tree_to_dict_estructura():
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    clf = DecisionTreeClassifier(max_depth=1).fit(X, y)
    d = tree_to_dict(clf, ["x"], ["no", "si"])
    assert "children_left" in d and "feature" in d and "threshold" in d
    assert d["feature_names"] == ["x"] and d["class_labels"] == ["no", "si"]
    assert len(d["children_left"]) == len(d["feature"])
