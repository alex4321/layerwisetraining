from typing import Dict
import pickle
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, path: str) -> None:
        with open(path, "rb") as f:
            self._data = pickle.load(f)
    
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        return self._data[index]
