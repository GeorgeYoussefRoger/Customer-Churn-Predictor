from sklearn.metrics import precision_score, recall_score, average_precision_score

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    pr_auc = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

    return {
        "Precision": precision,
        "Recall": recall,
        "PR-AUC": pr_auc
    }