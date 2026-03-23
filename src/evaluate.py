import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve

def evaluate(model, X_test, y_test):
    """
    Evaluate a trained model and return key metrics + optimal threshold.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Threshold tuning
    pr_precision, pr_recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (pr_precision * pr_recall) / (pr_precision + pr_recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "test_pr_auc": average_precision_score(y_test, y_proba),
        "threshold": best_threshold,
    }