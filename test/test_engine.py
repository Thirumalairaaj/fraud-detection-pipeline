from src.engine import trainEngine
from src.features import addFeatures, splitFeatures 

def test_trained_engine_tuned_threshold(labelledData):
    engine,_ = trainEngine(labelledData,n_splits=3)
    threshold = float(engine.best_threshold_)

    assert 0.0 < threshold < 1.0

    df = addFeatures(labelledData)
    X,_ = splitFeatures(df)
    proba = engine.predict_proba(X)[:, 1]
    pred = engine.predict(X)

    assert (pred == (proba >= threshold).astype(int)).all()

def test_trained_engine_scores(labelledData):
    _,metrics = trainEngine(labelledData,n_splits=3)
    assert (metrics["tn"]+metrics["fp"]+metrics["fn"]+metrics["tp"]) == 80
    assert metrics["roc_auc"] > 0.9