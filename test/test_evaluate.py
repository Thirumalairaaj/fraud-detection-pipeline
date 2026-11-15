import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 

from src.features import addFeatures, splitFeatures,scaleFeatures
from src.evaluate import plotPRCurve, createThresholdTable, bestf1Threshold, evaluateModel 


def test_plot_pr_curve():
    y_true = pd.Series([0,0,1,1])

    scores = {
        "model 1": np.array([0.1,0.4,0.35,0.8]),
        "model 2": np.array([0.2,0.3,0.6,0.9])
    }

    fig = plotPRCurve(scores,y_true)
    ax = fig.axes[0]

    assert len(ax.lines) == 3
    labels = [i.get_text() for i in ax.get_legend().get_texts()]

    assert any("model 1 PR-AUC" in label for label in labels)
    plt.close(fig)

def test_best_f1_threshold():
    y_true = np.array([0,0,1,1])
    y_score = np.array([0.1,0.4,0.35,0.8])

    result = bestf1Threshold(y_true,y_score)

    assert type(result) is float
    assert result == pytest.approx(0.35)

def test_create_threshold_table():
    y_true = np.array([0,0,1,1,1])
    y_score = np.array([0.1,0.6,0.4,0.7,0.9])
    thresholds = [0.5]

    table = createThresholdTable(y_true,y_score,thresholds)
    row = table.iloc[0]
    
    for col in ["tn","fp","fn","tp"]:
        assert pd.api.types.is_integer_dtype(table[col])

    assert (row["tn"],row["fp"],row["fn"],row["tp"]) == (1,1,1,2)

def test_evaluate_model(raw_df):
    raw_df = addFeatures(raw_df)
    X,y = splitFeatures(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)

    model = Pipeline(
        [
            ("preprocess",scaleFeatures()),
            ("clf",LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42))
        ]
    )

    model.fit(X_train,y_train)
    result = evaluateModel(model,X_test,y_test)

    y_true = y_test.to_numpy()
    pred = model.predict(X_test)
    #pred_proba =model.predict_prob(X_test)[:,1]

    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tp = int(((y_true == 1) & (pred == 1)).sum())

    assert (result["tn"],result["fp"],result["fn"],result["tp"]) == (tn,fp,fn,tp)
    assert result["tn"]+result["fp"]+result["fn"]+result["tp"] == len(y_test)

    
