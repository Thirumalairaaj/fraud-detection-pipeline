#preprocessing the dataset

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.clean import TARGET, cleanData, loadData

V_COLUMNS = [f"V{i}" for i in range(1,29)]
SCALE_COLUMNS = ["Amount_log", "Hour"]
FEATURE_COLUMNS = V_COLUMNS + SCALE_COLUMNS

def addFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """adding features to dataset copy"""

    df_copy = df.copy()
    df_copy["Amount_log"] = np.log1p(df_copy["Amount"])

    df_copy["Hour"] = (df_copy["Time"] // 3600) % 24

    return df_copy

def splitFeatures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """splitting features and targe"""

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

