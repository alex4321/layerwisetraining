import os
import hashlib
from typing import List
import random
import requests
import torch
import gzip
from datasets import load_dataset, Dataset, DatasetDict
from transformers.tokenization_utils import PreTrainedTokenizer


DOLMA_URLS = "https://huggingface.co/datasets/allenai/dolma/raw/main/urls/v1_7.txt"


def get_dolma_urls(seed: int) -> List[str]:
    """
    Fetch and shuffle Dolma URLs using a fixed random seed
    
    Args:
        seed (int): Random seed for shuffling
        
    Returns:
        list: Shuffled list of URLs as strings
    """
    # Fetch raw URL content
    response = requests.get(DOLMA_URLS)
    assert response.status_code == 200
    urls = [line.strip() for line in response.text.splitlines() if line.strip()]
    random.seed(seed)
    random.shuffle(urls)
    return urls


def decompress_n_layer_gzip(data: bytes) -> bytes:
    while data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    return data


def download_and_decompress(url: str, fname_full: str) -> str:
    if not os.path.exists(fname_full):
        tmp_path = fname_full + ".gz"
        if not os.path.exists(tmp_path):
            response = requests.get(url)
            response.raise_for_status()
            with open(tmp_path, "wb") as dst:
                dst.write(response.content)
        with open(tmp_path, "rb") as src:
            data = decompress_n_layer_gzip(src.read())
        with open(fname_full, "wb") as dst:
            dst.write(data)
    return fname_full


def download_dolma_archive(url: str, download_path: str) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()
    fname = f"{url_hash}.jsonl"
    fname_full = os.path.join(download_path, fname)
    for other_fname in os.listdir(download_path):
        if not other_fname.startswith(fname):
            os.remove(os.path.join(download_path, other_fname))
    if not os.path.exists(fname_full):
        download_and_decompress(url, fname_full)
    return fname_full


def load_dolma_dataset(fname: str) -> Dataset:
    """
    Load JSONL dataset from `fname` file using huggingface's datasets library
    Leave only the text field
    
    Args:
        fname (str): Path to the JSONL file
        
    Returns:
        Dataset: Huggingface dataset containing only the text field
    """
    dataset = load_dataset("json", data_files=fname)["train"]
    return dataset.select_columns(["text"])


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: int) -> Dataset:
    """
    Parallel batched tokenization of dataset using the provided tokenizer.
    Splits text into chunks and processes them in parallel for efficiency.
    
    Args:
        dataset (Dataset): Input dataset containing text field
        tokenizer (PreTrainedTokenizer): Tokenizer to use
        max_length (int): Maximum sequence length for tokenization
        
    Returns:
        Dataset: Tokenized dataset with input_ids and attention_mask
    """
    chunk_size = 10240  # Process texts in chunks of this size
    
    def tokenize_function(examples):
        encoded = tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = encoded["input_ids"]
        input_ids_tensors = [
            torch.LongTensor(item)
            for item in input_ids
        ]
        return {
            "input_ids": input_ids_tensors,
            "length": [len(item) for item in input_ids]
        }
    
    return dataset.map(
        tokenize_function,
        batched=True,
        batch_size=chunk_size,
        remove_columns=["text"]
    )


def shuffle_length_groups(dataset: Dataset, minibatch_size: int, random_seed: int) -> Dataset:
    """
    Shuffle dataset into minibatches of roughly equal length.
    """
    df_lengths = dataset.select_columns('length').to_pandas()
    df_lengths["index"] = list(range(len(df_lengths)))
    df_lengths = df_lengths.sort_values("length", ascending=False)
    df_lengths["group_id"] = [
        i // minibatch_size
        for i in range(len(dataset))
    ]
    unique_group_ids = df_lengths["group_id"].drop_duplicates()
    unique_group_ids = unique_group_ids.sample(n=len(unique_group_ids), random_state=random_seed)
    indices = df_lengths.set_index("group_id").loc[unique_group_ids.values, "index"].values
    return dataset.select(indices).select_columns(["input_ids"])


def train_test_split(dataset: Dataset, test_size: float, random_seed: int) -> DatasetDict:
    """
    Wrapper of transformers train/test split
    """
    dataset_dict = dataset.train_test_split(test_size=test_size, seed=random_seed)
    return dataset_dict