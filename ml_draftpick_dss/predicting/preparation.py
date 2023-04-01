import numpy as np
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

def normalize(df):
    scaler = SymmetricScaler().fit(df[NORMALIZE_COLS])
    df[NORMALIZED_COLS] = scaler.transform(df[NORMALIZE_COLS])
    return scaler

def calc_objective(target):
    target["objective"] = target["left_victory"] + (target["scores_sum_diff_norm"] / (2 + target["match_duration_norm"]))
    return target["objective"]

def extract_target(df):
    return torch.Tensor(df[TARGET_COLS].to_numpy().astype(float))

def split_dataframe(df, points, rand=42):
    return np.split(
        df.sample(frac=1, random_state=rand), 
        [int(x*len(df)) for x in points]
    )
