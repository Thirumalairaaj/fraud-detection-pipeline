import numpy as np
import json
import pytest 

from src.engine import trainEngine 
from src.artifacts import saveEngine, loadEngine 
from src.features import addFeatures, splitFeatures, FEATURE_COLUMNS 

def test_trained_engine_reload(labelledData,tmp_path):
    engine, metrics = trainEngine(df=labelledData,n_splits=3)
    saveEngine(engine=engine,metrics=metrics,df=labelledData,name="testing_engine",source="Synthetic",models_dir=tmp_path)
    loaded_engine, meta = loadEngine(name="testing_engine",models_dir=tmp_path)

    df = addFeatures(labelledData)
    X, _ = splitFeatures(df)

    np.testing.assert_array_equal(engine.predict_proba(X),loaded_engine.predict_proba(X))
    np.testing.assert_array_equal(engine.predict(X),loaded_engine.predict(X))

    assert meta["threshold"] == pytest.approx(float(engine.best_threshold_))
    assert meta["features"] == FEATURE_COLUMNS
    assert meta["reference"]["rows"] == len(labelledData)
    assert len(meta["reference"]["feature_mean"]) == len(FEATURE_COLUMNS)


def test_load_engine_sklearn_version(labelledData,tmp_path):
    engine,metrics = trainEngine(df=labelledData,n_splits=3)
    saveEngine(engine=engine,metrics=metrics,df=labelledData,name="testing_engine",source="Synthetic",models_dir=tmp_path)
    meta_path = tmp_path / "testing_engine.json"

    meta = json.loads(meta_path.read_text())

    meta["sklearn_version"] = "1.0.0"

    meta_path.write_text(json.dumps(meta,indent=2))

    with pytest.raises(RuntimeError):
        loadEngine(name="testing_engine",models_dir=tmp_path)





