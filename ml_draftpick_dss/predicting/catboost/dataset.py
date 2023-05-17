from torch.utils.data import Dataset
from ..preparation import TARGET_COLS, extract_target
from ..result_loader import flip_results, merge_results
import numpy as np
import pandas as pd

class ResultDataset(Dataset):

    def __init__(self, df, encoder, flip=True, weight=1.0, target_cols=["left_victory"]):
        if flip:
            df = merge_results([df, flip_results(df)])
        self.df = df
        if "weight" not in self.df.columns:
            self.df["weight"] = weight
        self.df["weight"].fillna(weight, inplace=True)
        self.encoder = encoder
        self.target_cols = target_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        
        left = self.encoder.encode_batch(sample["left_heroes"], dtype=np.intc)
        right = self.encoder.encode_batch(sample["right_heroes"], dtype=np.intc)

        left = np.sum(left, axis=-2)
        right = np.sum(right, axis=-2)

        x = np.concatenate([left, right], axis=-1)
        x = pd.DataFrame(x)

        y = extract_target(sample, target_cols=self.target_cols)
        y = pd.DataFrame(y)

        #weights = torch.full(target.shape, self.weight)
        weights = sample["weight"]
        weights = pd.DataFrame(weights)

        return x, y, weights 
    
    @property
    def all(self):
        return self[self.df.index]
