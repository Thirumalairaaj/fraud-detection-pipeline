import logging
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, TunedThresholdClassifierCV 
from sklearn.pipeline import Pipeline 

from src.clean import loadData, cleanData 
from src.features import addFeatures, splitFeatures, scaleFeatures 
from src.evaluate import evaluateModel 

logger = logging.getLogger(__name__)

RANDOM_STATE = 42 

def buildEngine(scoring: str='f1', n_splits: int=5) -> TunedThresholdClassifierCV:
    """Wraps the model and scaleFeatures in a pipeline to get best threshold during the fit"""
    model = Pipeline(
        [
            ("preprocess",scaleFeatures()),
            ("classifier",RandomForestClassifier(n_estimators=100,class_weight="balanced_subsample",n_jobs=-1,
                                                 random_state=RANDOM_STATE))
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=RANDOM_STATE)

    return TunedThresholdClassifierCV(
        estimator=model,
        scoring=scoring,
        cv=cv,
        refit=True
    )

def trainEngine(df: pd.DataFrame, test_size: float=0.2, n_splits: int = 5):

    df = addFeatures(df)
    X, y = splitFeatures(df)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,stratify=y,random_state=RANDOM_STATE)

    engine = buildEngine(n_splits=n_splits)
    engine.fit(X_train,y_train)
    logger.info("Tuned Threshold: %.4f ",engine.best_threshold_)

    metrics = evaluateModel(engine,X_test,y_test)
    return engine,metrics

if __name__ == "__main__":
    import sys 

    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/creditcard.csv"

    data = loadData(path)
    cleaned_data = cleanData(data)
    engine,metrics = trainEngine(cleaned_data)
    print("cross-validated F1 at that threshold:", round(float(engine.best_score_), 4))
    print(pd.Series(metrics).round(4))
