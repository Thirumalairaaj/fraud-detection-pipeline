#Evaluation metrics for the fraud classifiers

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

def evaluateModel(model, X_test, y_test) -> dict:
    """returns key metrics for the model evaluation"""

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]
    tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()

    metrics = {
        "precision": precision_score(y_test,y_pred),
        "recall": recall_score(y_test,y_pred),
        "f1": f1_score(y_test,y_pred),
        "pr_auc": average_precision_score(y_test,y_score),
        "roc_auc": roc_auc_score(y_test,y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    return metrics

