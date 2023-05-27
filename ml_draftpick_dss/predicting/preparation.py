import numpy as np
import pandas as pd
from .result_loader import NORMALIZE_COLS, NORMALIZED_COLS
import torch

TARGET_COLS=["left_victory", "scores_sum_diff_norm", "match_duration_norm"]

class SymmetricScaler:
    def __init__(self):
        pass

    def fit(self, X):
        mean = X.mean(axis=0)
        minmax = X.min(axis=0), X.max(axis=0)
        diff = [abs(m-mean) for m in minmax]
        max_diff = np.maximum(*diff)
        self.mean = mean
        self.max_diff = max_diff
        return self

    def transform(self, X):
        X_std = (X - self.mean) / self.max_diff
        return X_std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_std):
        X = X_std * self.max_diff + self.mean
        return X

def normalize(df, scaler=None):
    if not scaler:
        scaler = SymmetricScaler().fit(df[NORMALIZE_COLS])
    df[NORMALIZED_COLS] = scaler.transform(df[NORMALIZE_COLS])
    return scaler

def calc_objective(target):
    target["objective"] = target["left_victory"] + (target["scores_sum_diff_norm"] / (2 + target["match_duration_norm"]))
    return target["objective"]

def extract_target(df, target_cols=TARGET_COLS):
    return torch.Tensor(df[target_cols].to_numpy().astype(float))

def split_dataframe(df, points, rand=42):
    return np.split(
        df.sample(frac=1, random_state=rand), 
        [int(x*len(df)) for x in points]
    )

def split_dataframe_kfold(df, ratio=0.2, rand=42, val=True, filter_i=None):
    result = []
    count = int(1.0/ratio)
    splits = [i*ratio for i in range(1, count)]
    splits = split_dataframe(df, splits, rand=rand)

    for i in range(count):
        if filter_i and i not in filter_i:
            continue
        j = count - 1 + i
        test_df = splits[j%count]
        val_df = None
        if val:
            val_df = splits[(j-1)%count]
        train_dfs = [s for s in splits if s is not test_df and s is not val_df]
        train_df = pd.concat(train_dfs)
        result.append((train_df, val_df, test_df))

    return result
