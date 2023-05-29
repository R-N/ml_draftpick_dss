from torch.utils.data import Dataset, DataLoader
from ..preparation import TARGET_COLS, extract_target
from ..result_loader import flip_results, merge_results
import torch
from ..encoding import HeroOneHotEncoder
from ..transformer.dataset import load_datasets as _load_datasets, load_datasets_kfold as __load_datasets_kfold, _load_datasets_kfold as ___load_datasets_kfold

class ResultDataset(Dataset):

    def __init__(self, df, encoder, flip=True, weight=1.0, target_cols=TARGET_COLS):
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
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.df.iloc[idx]
        
        left = self.encoder.encode_batch(sample["left_heroes"])
        right = self.encoder.encode_batch(sample["right_heroes"])

        left = torch.Tensor(left)
        right = torch.Tensor(right)

        left = torch.sum(left, dim=-2)
        right = torch.sum(right, dim=-2)

        target = extract_target(sample, target_cols=self.target_cols)

        #weights = torch.full(target.shape, self.weight)
        weights = sample["weight"]

        return left, right, target, weights 
    

def load_datasets(
    *args,
    encoder_factory=HeroOneHotEncoder,
    dataset_factory=ResultDataset,
    **kwargs
):
    return _load_datasets(
        *args,
        encoder_factory=encoder_factory,
        dataset_factory=dataset_factory,
        **kwargs
    )

def _load_datasets_kfold(
    *args,
    encoder_factory=HeroOneHotEncoder,
    dataset_factory=ResultDataset,
    **kwargs
):
    return ___load_datasets_kfold(
        *args,
        encoder_factory=encoder_factory,
        dataset_factory=dataset_factory,
        **kwargs
    )

def load_datasets_kfold(func=_load_datasets_kfold, **kwargs):
    return __load_datasets_kfold(func=func, **kwargs)

def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader