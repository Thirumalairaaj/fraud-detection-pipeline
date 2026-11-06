#model training

#import argparse
import logging
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.clean import loadData, cleanData
from src.evaluate import evaluateModel
from src.features import addFeatures, splitFeatures, scaleFeatures

logger = logging.getLogger(__name__)
DATA_PATH = "data/raw/creditcard.csv"

RANDOM_STATE = 58

def buildModels() -> dict:
    """returns dict of models"""

    models = {
        "logistic regression": Pipeline(
            [
                ("preprocess", scaleFeatures()),
                ("classifier", LogisticRegression(
                    class_weight = "balanced",
                    max_iter = 1000,
                    random_state = RANDOM_STATE 
                ))
            ]
        ),
        "random forest": Pipeline(
            [
                ("preprocess",scaleFeatures()),
                ("classifier", RandomForestClassifier(
                    n_estimators=100, class_weight="balanced_subsample",
                    n_jobs=-1, random_state= RANDOM_STATE,
                    bootstrap=False
                ))
            ]
        )
    }

    return models

def run() -> pd.DataFrame:
    df = loadData(DATA_PATH)
    df = cleanData(df)
    df = addFeatures(df)

    X,y = splitFeatures(df) #X - feature columns, y- target column

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=RANDOM_STATE)

    logger.info("Fraud rate : train set: %.4f, test set: %.4f, test fraud %d",y_train.mean()*100,y_test.mean()*100,
                y_test.sum())

    results = {}

    for name,model in buildModels().items():
        logger.info("Training: %s....",name)
        model.fit(X_train,y_train)
        results[name] = evaluateModel(model,X_test,y_test)

    table = pd.DataFrame(results).T 

    return table



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    print(run())
    
