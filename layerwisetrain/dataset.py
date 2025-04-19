from typing import Dict, List
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


def collate_fn(batch: List[Dict[str, torch.Tensor | List[int]]]) -> Dict[str, torch.Tensor]:
    input_sizes = []
    used_keys = None
    for item in batch:
        if used_keys is None:
            used_keys = set(item.keys())
        else:
            assert set(item.keys()) == used_keys
        if "input_ids" in item:
            input_sizes.append(torch.LongTensor(item["input_ids"]).size(0))
        elif "inputs_embeds" in item:
            input_sizes.append(torch.LongTensor(item["inputs_embeds"]).size(0))
    assert ("input_ids" in used_keys) ^ ("inputs_embeds" in used_keys), "Only one of input_ids or inputs_embeds should be present"
    max_size = max(input_sizes)
    pad_values = {
        "labels": -100,
        "attention_mask": 0,
        "position_ids": 0,
        "input_ids": 0,
        "inputs_embeds": 0,
    }
    batch_padded = {}
    for key, pad_value in pad_values.items():
        if key not in used_keys:
            continue
        values = [
            torch.LongTensor(item[key])
            for item in batch
        ]
        lengths = [
            value.shape[0]
            for value in values
        ]
        pad_sizes = [
            max_size - length
            for length in lengths
        ]
        values_padded = [
            torch.cat(
                [
                    value,
                    torch.ones(
                        (value_pad_size, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    ) * pad_value,
                ],
                dim=0,
            )
            for value, value_pad_size in zip(values, pad_sizes)
        ]
        batch_padded[key] = torch.stack(values_padded)
    return batch_padded