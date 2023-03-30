from torch.utils.data import Dataset, DataLoader
from ml_draftpick_dss.predicting.preparation import TARGET_COLS, extract_target
from ml_draftpick_dss.predicting.result_loader import flip_results, merge_results
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
            left = torch.LongTensor(left)
            right = torch.LongTensor(right)

        target_df = sample[TARGET_COLS]
        target = extract_target(target_df)

        return left, right, target