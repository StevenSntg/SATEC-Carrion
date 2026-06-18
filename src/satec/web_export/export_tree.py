"""Serializa un DecisionTreeClassifier de sklearn a un dict JSON plano."""


def tree_to_dict(clf, feature_names, class_labels) -> dict:
    t = clf.tree_
    return {
        "feature_names": list(feature_names),
        "class_labels": list(class_labels),
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": t.value.tolist(),
    }
