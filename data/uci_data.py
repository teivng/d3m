import os
import torch
from omegaconf import DictConfig

from .common import TensorDataset


# ================================================
# =========UCI Heart Disease Utilities============
# ================================================


class UCIDataset(TensorDataset):
    """UCI Heart Disease Dataset class.

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
    """
    def __init__(self, X:torch.tensor, y:torch.tensor):
        super(UCIDataset, self).__init__(X=X, y=y)
        
    def __str__(self):
        return f"""Dataset UCI Heart Disease
    \tNumber of datapoints: {self.__len__()}
    \tRoot location: data/uci_data/
    """
        

def get_uci_datasets(args:DictConfig) -> dict:
    """Returns processed UCI Heart Disease Dataset objects

    Args:
        args (DictConfig): hydra arguments

    Returns:
        dict: dictionary containing all splits of the UCI Heart Disease processed dataset.
    """
    data_path = os.path.join(args.dataset.data_dir, 'uci_heart_torch.pt')
    assert os.path.exists(data_path)        
    data_dict = torch.load(data_path)
    processed_dict = {}
    for k, data in data_dict.items():
        data = list(zip(*data))
        X, y = torch.stack(data[0]), torch.tensor(data[1], dtype=torch.int)
        if args.dataset.normalize:
            min_ = torch.min(X, dim=0).values
            max_ = torch.max(X, dim=0).values
            X = (X - min_) / (max_ - min_)
        processed_dict[k] = UCIDataset(X, y)
        
    return processed_dict