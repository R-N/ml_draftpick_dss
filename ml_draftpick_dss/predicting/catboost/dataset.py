from torch.utils.data import Dataset
from ..preparation import TARGET_COLS, extract_target
from ..result_loader import flip_results, merge_results
import numpy as np
import pandas as pd
from ..mlp.dataset import load_datasets as _load_datasets, load_datasets_kfold as __load_datasets_kfold, _load_datasets_kfold as ___load_datasets_kfold
from ..util import tanh_to_sig_range
from catboost import Pool
from ..encoding import HeroLabelEncoder, HeroOneHotEncoder, PATCHES

class ResultDataset(Dataset):

    def __init__(self, df, encoder, flip=True, weight=1.0, target_cols=["left_victory"]):
        if flip:
            df = merge_results([df, flip_results(df)])
        self.df = df.reset_index()
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
    
def _create_dataset(X, y, w, cat_ids):
    pool = Pool(X, y, cat_features=cat_ids, weight=w)
    pool.X = X
    pool.y = y
    pool.w = w
    return pool
    
def _create_datasets(*datasets):
    datasets = [[x.all for x in ds] for ds in datasets]
    datasets = [tuple(pd.concat(x) for x in zip(*ds)) for ds in datasets]
    datasets = [(X, tanh_to_sig_range(y), w) for X, y, w in datasets]
    cat_ids = np.where(datasets[0][0].dtypes != float)[0]
    datasets = [_create_dataset(X, y, w, cat_ids) for X, y, w in datasets]
    return datasets

def load_datasets(
    *args,
    create_datasets=_create_datasets,
    encoder_factory=HeroLabelEncoder,
    dataset_factory=ResultDataset,
    **kwargs
):
    return _load_datasets(
        *args,
        create_datasets=create_datasets,
        encoder_factory=encoder_factory,
        dataset_factory=dataset_factory,
        **kwargs
    )

def _load_datasets_kfold(
    *args,
    create_datasets=_create_datasets,
    encoder_factory=HeroOneHotEncoder,
    dataset_factory=ResultDataset,
    **kwargs
):
    return ___load_datasets_kfold(
        *args,
        create_datasets=create_datasets,
        encoder_factory=encoder_factory,
        dataset_factory=dataset_factory,
        **kwargs
    )

def load_datasets_kfold(func=_load_datasets_kfold, **kwargs):
    return __load_datasets_kfold(func=func, **kwargs)

def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=0):
    return dataset
