import pytest
import pandas as pd
import numpy as np

from src.engine import trainEngine
from src.predict import predictProba

@pytest.fixture
def engineWithData(labelledData):

    engine,_ = trainEngine(labelledData,n_splits=3)

    return engine,labelledData 

def test_predict_empty_input(engineWithData):
    
    engine, df = engineWithData 

    assert len(predictProba(engine,df.iloc[0:0])) == 0

def test_predict_order(engineWithData):

    engine,df = engineWithData 

    result = predictProba(engine,df)

    assert list(result.index) == list(df.index)
    assert list(result.columns) == ["Fraud probability","Predicted class"]
    assert result["Fraud probability"].between(0,1).all()

def test_predict_engine_threshold(engineWithData):
    
    engine,df = engineWithData

    result = predictProba(engine,df)

    expected = (result["Fraud probability"] >= engine.best_threshold_).astype(int)

    assert (result["Predicted class"] == expected).all()

def test_predict_duplicates(engineWithData):
    
    engine,df = engineWithData

    dups = pd.concat([df, df.iloc[[0,0,0]]],ignore_index=True)

    result = predictProba(engine,dups)

    assert len(result) == len(dups)

def test_predict_accepts_extra_columns(engineWithData):
    
    engine, df = engineWithData 

    df_copy = df.drop(columns=["Class"]).copy()
    df_copy.insert(0,"id",range(len(df_copy)))
    df_copy["Time"] = 1.0

    result = predictProba(engine,df_copy)

    assert len(result) == len(df_copy)

def test_predict_refuse_invalid_data(engineWithData):
    
    engine,df = engineWithData

    for bad in (df.drop(columns=["V3"]),df.assign(Amount= -200.0),df.assign(Amount=np.nan)):
        with pytest.raises(ValueError):
            predictProba(engine,bad)




