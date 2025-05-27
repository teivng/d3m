from typing import Tuple
import torch
from abc import ABC, abstractmethod

from torch.utils.data import Dataset, random_split


# ================================================
# ===========General Data Utilities===============
# ================================================


def split_dataset(dataset: Dataset, lengths: list, random_seed: int = 57) -> Tuple[Dataset, Dataset]:
    """Splits the dataset into chunks specified by lengths.

    Args:
        dataset (Dataset): dataset to be splitted
        lengths (list): split specifications, see torch.utils.data.random_split 
        random_seed (int, optional): Defaults to 57.

    Returns:
        Tuple[Dataset, ...]: splitted datasets
    """
    return random_split(dataset, lengths,
                        generator=torch.Generator().manual_seed(random_seed))


class TensorDataset(Dataset, ABC):
    """Abstract torch tensor dataset class.
    Requires an implementation of TensorDataset.__str__

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self) -> int:
        return len(self.y)
    
    @abstractmethod
    def __str__(self) -> str:
        pass