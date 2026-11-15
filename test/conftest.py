import matplotlib 
import numpy as np
import pandas as pd
import random
import pytest

matplotlib.use("Agg")

V_COLUMNS = [f"V{i}"for i in range(1,29)]
m = len(V_COLUMNS)
class_col = [0] * 20 + [1] * 5
random.shuffle(class_col)

@pytest.fixture 
def raw_df() -> pd.DataFrame:
    n = 25
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n,m)),columns=V_COLUMNS)
    df.insert(0,"Time",np.arange(n)*1000.0)
    df["Amount"] = rng.uniform(0,500,n)
    df["Class"] = class_col 

    return df