import os
import pandas as pd
from pathlib import Path

def mkdir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def df_to_dict(df, key, value):
    return pd.Series(df[key].values,index=df[value]).to_dict()

def list_subdirs(dir):
    return [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
