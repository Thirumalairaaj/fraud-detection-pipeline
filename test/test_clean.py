import pandas as pd
import pytest

from src.clean import loadData, cleanData 

def test_load_data_file_exists_check(tmp_path):

    with pytest.raises(FileNotFoundError):
        loadData(tmp_path / "creditcard.csv")

def test_clean_data_missing_col_check(raw_df):

    raw_df = raw_df.drop(columns=["V5"])

    with pytest.raises(ValueError):
        cleanData(raw_df)

def test_clean_data_missing_val_check(raw_df):

    raw_df.loc[5,"Amount"] = None

    with pytest.raises(ValueError):
        cleanData(raw_df)

def test_clean_data_class_col_check(raw_df):

    raw_df.loc[3,"Class"] = 12

    with pytest.raises(ValueError):
        cleanData(raw_df)

def test_clean_data_amount_col_check(raw_df):

    raw_df.loc[5,"Amount"] = -243

    with pytest.raises(ValueError):
        cleanData(raw_df)

def test_clean_data_time_col_check(raw_df):

    raw_df.loc[7,"Time"] = -1200

    with pytest.raises(ValueError):
        cleanData(raw_df)

def test_clean_data_duplicate_remove_check(raw_df):
    with_dup = pd.concat([raw_df.iloc[:5],raw_df.iloc[[3]],raw_df.iloc[5:]], ignore_index=True)
    result = cleanData(with_dup)

    assert len(result) == len(raw_df)
    assert list(result.index) == list(range(len(result)))

def test_clean_data_valid_data_check(raw_df):
    result = cleanData(raw_df)

    assert len(result) == len(raw_df)