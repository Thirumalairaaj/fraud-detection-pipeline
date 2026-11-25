#Evaluation metrics for the fraud classifiers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve
)

def plotPRCurve(score: dict, y_true):
    """plots precision-recall curve"""

    fig,ax = plt.subplots(figsize=(7,4))

    for name,score in score.items():
        precision, recall, _ = precision_recall_curve(y_true,score)
        average_precision = average_precision_score(y_true,score)
        ax.step(recall,precision,where="post",label=f"{name} PR-AUC = {average_precision:.3f}")

    ax.axhline(y_true.mean(),linestyle="--", color="grey", label = "no-skill baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()

    return fig

def createThresholdTable(y_true, y_score,thresholds) -> pd.DataFrame:
    """Metrics at each decision threshold"""

    rows = []
    for i in thresholds:
        y_pred = (y_score >= i).astype(int)
        tn,fp,fn,tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

        rows.append(
            {
                "Threshold": round(float(i),3),
                "Precision": precision_score(y_true,y_pred,zero_division=0),
                "Recall": recall_score(y_true,y_pred, zero_division=0),
                "f1": f1_score(y_true,y_pred,zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        )

    return pd.DataFrame(rows)


def bestf1Threshold(y_true,y_score) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true,y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return float(thresholds[np.argmax(f1[:-1])])

def evaluateModel(model, X_test, y_test) -> dict:
    """returns key metrics for the model evaluation"""

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:,1]
    tn,fp,fn,tp = confusion_matrix(y_test,y_pred,labels=[0,1]).ravel()

    metrics = {
        "precision": precision_score(y_test,y_pred,zero_division=0),
        "recall": recall_score(y_test,y_pred,zero_division=0),
        "f1": f1_score(y_test,y_pred,zero_division=0),
        "pr_auc": average_precision_score(y_test,y_score),
        "roc_auc": roc_auc_score(y_test,y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    return metrics

