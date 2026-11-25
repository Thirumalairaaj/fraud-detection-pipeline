import json
from datetime import datetime,timezone
from pathlib import Path 

import joblib
import pandas as pd
from sklearn import __version__
from src.features import FEATURE_COLUMNS, addFeatures, selectFeatures

MODELS_DIR = Path("models")

def saveEngine(engine,metrics: dict, df: pd.DataFrame, name: str, source: str="", 
               models_dir: Path=MODELS_DIR) -> tuple[Path,Path]:

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True,exist_ok=True)
    df = addFeatures(df)
    X = selectFeatures(df)

    meta = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source,
        "features": FEATURE_COLUMNS,
        "threshold": float(engine.best_threshold_),
        "cv_score": float(engine.best_score_),
        "test_metrics": {k: float(v) for k,v in metrics.items()},
        "sklearn_version": __version__,
        "reference":{
            "rows": len(df),
            "fraud_share": float(df["Class"].mean()),
            "feature_mean": {c: float(v) for c, v in X.mean().items()},
            "feature_std": {c: float(v) for c, v in X.std().items()}
        }
    }

    model_path = models_dir / f"{name}.joblib"
    meta_path = models_dir / f"{name}.json"

    joblib.dump(engine,model_path,compress=3)
    meta_path.write_text(json.dumps(meta, indent=2))

    return model_path,meta_path

def loadEngine(name: str, models_dir: Path=MODELS_DIR):
    models_dir = Path(models_dir)
    meta = json.loads((models_dir / f"{name}.json").read_text())

    if meta["sklearn_version"] != __version__:
        raise RuntimeError(
            f"Model {name} was saved with scikit-learn : {meta["sklearn_version"]},"
            f"but this environment has scikit-lear: {__version__}; Re-train the model"
        )

    engine = joblib.load(models_dir / f"{name}.joblib")

    return engine, meta