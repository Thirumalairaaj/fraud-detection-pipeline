import sys 
import pandas as pd

from src.artifacts import loadEngine
from src.clean import loadData, validateFeatures
from src.features import addFeatures, selectFeatures


def predictProba(engine,df: pd.DataFrame) -> pd.DataFrame:

    if len(df) == 0:
        return pd.DataFrame({"Fraud probability": [],"Predicted class": []},index=df.index)

    validateFeatures(df)
    df = addFeatures(df)
    X = selectFeatures(df)

    result = pd.DataFrame(
        {
            "Fraud probability": engine.predict_proba(X)[:,1],
            "Predicted class": engine.predict(X)
        },
        index=df.index
    )

    return result


if __name__  == "__main__":

    name,in_path,out_path = sys.argv[1:4]
    engine,meta = loadEngine(name=name)

    df = loadData(in_path)
    result = predictProba(engine,df=df)

    if "id" in df.columns:
        result.insert(0,"id",df["id"])

    result.to_csv(out_path,index=False)
    print(f"Scored {len(result)} rows with {name}; flagged {int(result['Predicted class'].sum())} as fraud")

    