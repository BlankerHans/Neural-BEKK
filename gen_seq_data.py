# generate sequencial data
import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx+self.lookback]
        y_t   = self.y[idx+self.lookback]  # next step
        return torch.from_numpy(x_seq), torch.from_numpy(np.array(y_t))