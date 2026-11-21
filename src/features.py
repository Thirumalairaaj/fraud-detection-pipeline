#preprocessing the dataset

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.clean import TARGET, V_COLUMNS,cleanData, loadData

SCALE_COLUMNS = ["Amount_log"]
FEATURE_COLUMNS = V_COLUMNS + SCALE_COLUMNS

def addFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """adding features to dataset copy"""

    df_copy = df.copy()
    df_copy["Amount_log"] = np.log1p(df_copy["Amount"])

    return df_copy

def selectFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """For API call. Model inputs in fixed order"""

    return df[FEATURE_COLUMNS]

def splitFeatures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """splitting features and target"""

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    return X,y

def scaleFeatures() -> ColumnTransformer:
    """Scaling columns"""
    col_transformer = ColumnTransformer(
        transformers = [("scale", StandardScaler(), SCALE_COLUMNS)],
        remainder = "passthrough"
    )

    return col_transformer

if __name__ == "__main__":
    df = cleanData(loadData("data/raw/creditcard.csv"))
    df = addFeatures(df)
    X,y = splitFeatures(df)
    print("X shape: ", X.shape," y shape: ",y.shape)
    print(X[SCALE_COLUMNS].describe().round(2))

