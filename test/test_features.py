import pandas as pd
import numpy as np


from src.features import SCALE_COLUMNS, addFeatures, splitFeatures, scaleFeatures

def test_add_features(raw_df):
    raw_copy = raw_df.copy()
    result = addFeatures(raw_df)

    assert {"Amount_log","Hour"} <= set(result.columns)
    pd.testing.assert_frame_equal(raw_df,raw_copy) #to check addFeatures() doesn't affect original data. Any modification happens it should effect on copy of orginal data

def test_add_features_amount_col_conversion(raw_df):
    result = addFeatures(raw_df)

    np.testing.assert_allclose(result["Amount_log"],np.log1p(raw_df["Amount"]))

def test_add_features_time_col_conversion(raw_df):
    raw_df.loc[0,"Time"] = 0
    raw_df.loc[1,"Time"] = 3600 * 25 #25th hour 90,000 seconds

    result = addFeatures(raw_df)

    assert result.loc[0,"Hour"] == 0
    assert result.loc[1,"Hour"] == 1
    assert result["Hour"].between(0,23).all()

def test_split_features(raw_df):
    raw_df = addFeatures(raw_df)
    X,y = splitFeatures(raw_df)

    assert "Class" not in X.columns 
    assert len(X) == len(y) == len(raw_df)
    assert set(y.unique()) <= {0,1}

def test_scale_features_scale_col_check(raw_df):
    raw_df = addFeatures(raw_df)
    X,_ = splitFeatures(raw_df)
    n = len(SCALE_COLUMNS)

    result = scaleFeatures().fit_transform(X)

    np.testing.assert_allclose(result[:,:n].mean(axis=0),0,atol=1e-9)
    np.testing.assert_allclose(result[:,:n].std(axis=0),1,atol=1e-9)
    np.testing.assert_allclose(result[:,n:],X.drop(columns=SCALE_COLUMNS).to_numpy())

def test_scale_features_data_leakage(raw_df):
    raw_df = addFeatures(raw_df)
    X,_ = splitFeatures(raw_df)
    n = len(SCALE_COLUMNS)

    train,test = X.iloc[:15], X.iloc[15:]
    fits = scaleFeatures().fit(train)
    result = fits.transform(test)

    expected = (test[SCALE_COLUMNS] - train[SCALE_COLUMNS].mean()) / train[SCALE_COLUMNS].std(ddof=0)

    np.testing.assert_allclose(result[: , :n],expected.to_numpy())