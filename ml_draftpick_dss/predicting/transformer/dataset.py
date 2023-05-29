from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import pandas as pd
import os
from ..preparation import TARGET_COLS, extract_target
from ..result_loader import flip_results, merge_results, load_victory
from ..preparation import normalize
from ..heroes_loader import load_heroes
from ..encoding import HeroLabelEncoder, PATCHES
from ..preparation import split_dataframe, split_dataframe_kfold

class ResultDataset(Dataset):

    def __init__(self, df, encoder, embedder=None, flip=True, weight=1.0, target_cols=TARGET_COLS):
        if flip:
            df = merge_results([df, flip_results(df)])
        self.df = df
        if "weight" not in self.df.columns:
            self.df["weight"] = weight
        self.df["weight"].fillna(weight, inplace=True)
        self.encoder = encoder
        self.embedder = embedder
        self.target_cols = target_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.df.iloc[idx]
        
        left = self.encoder.encode_batch(sample["left_heroes"])
        right = self.encoder.encode_batch(sample["right_heroes"])

        if self.embedder:
            left = self.embedder.embed_batch(left)
            right = self.embedder.embed_batch(right)
        else:
            left = torch.Tensor(left)
            right = torch.Tensor(right)

        left = shuffle_hero(left)
        right = shuffle_hero(right)

        target = extract_target(sample, target_cols=self.target_cols)

        #weights = torch.full(target.shape, self.weight)
        weights = sample["weight"]

        return left, right, target, weights 
    
def shuffle_hero(t):
    return t[..., torch.randperm(t.shape[-2]), :]

def load_datasets(
    patches=PATCHES,
    result_file="results.csv",
    heroes_file="heroes.csv",
    data_dir="../csv",
    weights=None,
    ratio=0.2,
    encoder_factory=HeroLabelEncoder,
    dataset_factory=ResultDataset,
    create_datasets=None
):

    dfs = []
    for patch in patches:
        result_path = os.path.join(data_dir, patch, result_file)
        _df = load_victory(result_path)
        _df["patch"] = patch
        if weights:
            _df["weight"] = weights[patch]
        dfs.append(_df)
        
    df = pd.concat(dfs)
    scaler = normalize(df)

    _ = [normalize(df, scaler=scaler) for df in dfs]

    encoders = []
    train_dfs, val_dfs, test_dfs = [], [], []
    train_sets, val_sets, test_sets = [], [], []
    encoder = None

    for _df, patch in zip(dfs, patches):
        heroes_path = os.path.join(data_dir, patch, heroes_file)
        df_heroes = load_heroes(heroes_path)
        df_heroes["patch"] = patch
        if weights:
            df_heroes["weight"] = weights[patch]
        encoder = encoder_factory(df_heroes)
        encoders.append(encoder)

        train_df, val_df, test_df = _dfs = split_dataframe(_df, (1-ratio*2, 1-ratio*1), rand=42)
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

        train_set, val_set, test_set = _datasets = [dataset_factory(__df, encoder) for __df in _dfs]
        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    if create_datasets:
        train_set, val_set, test_set = create_datasets(train_sets, val_sets, test_sets)
    else:
        train_set = ConcatDataset(train_sets)
        val_set = ConcatDataset(val_sets)
        test_set = ConcatDataset(test_sets)

    return {
        "df": df,
        "dfs": dfs,
        "encoders": encoders,
        "split_dfs": (train_df, val_df, test_df),
        "split_datasets": (train_set, val_set, test_set)
    }

def _load_datasets_kfold(
    i=0,
    patches=PATCHES,
    result_file="results.csv",
    heroes_file="heroes.csv",
    data_dir="../csv",
    weights=None,
    ratio=0.2,
    val=True,
    encoder_factory=HeroLabelEncoder,
    dataset_factory=ResultDataset,
    create_datasets=None
):

    dfs = []
    for patch in patches:
        result_path = os.path.join(data_dir, patch, result_file)
        _df = load_victory(result_path)
        _df["patch"] = patch
        if weights:
            _df["weight"] = weights[patch]
        dfs.append(_df)
        
    df = pd.concat(dfs)
    scaler = normalize(df)

    _ = [normalize(df, scaler=scaler) for df in dfs]

    encoders = []
    train_dfs, val_dfs, test_dfs = [], [], []
    train_sets, val_sets, test_sets = [], [], []
    encoder = None

    for _df, patch in zip(dfs, patches):
        heroes_path = os.path.join(data_dir, patch, heroes_file)
        df_heroes = load_heroes(heroes_path)
        df_heroes["patch"] = patch
        if weights:
            df_heroes["weight"] = weights[patch]
        encoder = encoder_factory(df_heroes)
        encoders.append(encoder)

        train_df, val_df, test_df = _dfs = split_dataframe_kfold(_df, ratio=ratio, val=val, filter_i={i})[0]
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

        train_set, val_set, test_set = _datasets = [dataset_factory(__df, encoder) for __df in _dfs]
        train_sets.append(train_set)
        val_sets.append(val_set)
        test_sets.append(test_set)

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    if create_datasets:
        train_set, val_set, test_set = create_datasets(train_sets, val_sets, test_sets)
    else:
        train_set = ConcatDataset(train_sets)
        val_set = ConcatDataset(val_sets)
        test_set = ConcatDataset(test_sets)

    return {
        "df": df,
        "dfs": dfs,
        "encoders": encoders,
        "split_dfs": (train_df, val_df, test_df),
        "split_datasets": (train_set, val_set, test_set)
    }

def load_datasets_kfold(ratio=0.2, func=_load_datasets_kfold, **kwargs):
    result = []
    count = int(1.0/ratio)
    for i in range(count):
        r = func(i=i, ratio=ratio, **kwargs)
        result.append(r)
    return result

def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
