#Data cleansing

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "Class"
EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount", TARGET]

def loadData(path) -> pd.DataFrame:
    """ loads the data from given path and converts csv into pandas dataframe"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded %d rows and %d columns from %s", *df.shape,path)

    return df

def validateSchema(df: pd.DataFrame) -> None:
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in the dataset: {missing_columns}")

    if df[EXPECTED_COLUMNS].isnull().any().any():
        raise ValueError("Dataset contains missing values")

    if not (set(df["Class"].unique()) <= {0,1}):
        raise ValueError("Class should only contain either 0 or 1")

    if df["Amount"].min() < 0:
        raise ValueError("Amount should not be negative")

    if df["Time"].min() < 0:
        raise ValueError("Time should not be negative")

def dropDuplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d duplicate rows",before - len(df))

    return df

def cleanData(df: pd.DataFrame) -> pd.DataFrame:
    validateSchema(df)

    return dropDuplicates(df)


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")
    raw = loadData("data/raw/creditcard.csv")
    cleaned = cleanData(raw)
    print("cleaned data:",cleaned.shape)
    print(cleaned[TARGET].value_counts())

