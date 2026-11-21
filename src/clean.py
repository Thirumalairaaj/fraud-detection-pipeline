#Data cleansing

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "Class"
V_COLUMNS = [f"V{i}" for i in range(1,29)]
FEATURE_COLUMNS = V_COLUMNS + ["Amount"]
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET]

def loadData(path) -> pd.DataFrame:
    """ loads the data from given path and converts csv into pandas dataframe"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded %d rows and %d columns from %s", *df.shape,path)

    return df

def validateFeatures(df: pd.DataFrame) -> None:
    """Checks for any transactions. Also, Used by API"""

    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if df[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Contains Missing values")

    if df["Amount"].min() < 0:
        raise ValueError("Amount should not be negative")

def validateLabelled(df: pd.DataFrame) -> None:
    validateFeatures(df)
    if TARGET not in df.columns:
        raise ValueError(f"Missing column in the dataset: {TARGET}")

    if df[TARGET].isnull().any():
        raise ValueError("Dataset contains missing values")

    if not (set(df[TARGET].unique()) <= {0,1}):
        raise ValueError("Class should only contain either 0 or 1")


def dropDuplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=EXPECTED_COLUMNS).reset_index(drop=True)
    logger.info("Dropped %d duplicate rows",before - len(df))

    return df

def cleanData(df: pd.DataFrame) -> pd.DataFrame:
    validateLabelled(df)

    return dropDuplicates(df)


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")
    raw = loadData("data/raw/creditcard.csv")
    cleaned = cleanData(raw)
    print("cleaned data:",cleaned.shape)
    print(cleaned[TARGET].value_counts())

