from torch.utils.data import Dataset, DataLoader
from ..preparation import TARGET_COLS, extract_target
from ..result_loader import flip_results, merge_results
import torch

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

def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader