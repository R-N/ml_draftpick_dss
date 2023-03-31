from torch.utils.data import Dataset, DataLoader
from .preparation import TARGET_COLS, extract_target
from .result_loader import flip_results, merge_results
import torch

class ResultDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, encoder, embedder=None, flip=True):
        if flip:
            df = merge_results([df, flip_results(df)])
        self.df = df
        self.encoder = encoder
        self.embedder = embedder

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

        target_df = sample[TARGET_COLS]
        target = extract_target(target_df)

        return left, right, target
    
def shuffle_hero(t):
    return t[..., torch.randperm(t.shape[-2]), :]

def create_dataloader(df, encoder, embedder=None, batch_size=32, shuffle=True, num_workers=0):
    dataset = ResultDataset(df, encoder, embedder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader