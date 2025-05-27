import numpy as np
import torch
from omegaconf import DictConfig
from .common import TensorDataset


# ================================================
# ===========Synthetic Data Utilities=============
# ================================================


class SyntheticDataset(TensorDataset):
    """Synthetic Dataset class

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
    """
    def __init__(self, X:torch.tensor, y:torch.tensor):
        super(SyntheticDataset, self).__init__(X=X, y=y)    

    def __str__(self):
        return f"""Dataset Synthetic
    \tNumber of datapoints: {self.__len__()}
    \tRoot location: data/synthetic_data/
    """
    
def get_synthetic_datasets(args:DictConfig) -> dict:
    """Returns synthetic data according to generation parameters

    Args:
        args (DictConfig): hydra arguments

    Returns:
        dict: dictionary containing all splits of generated synthetic data.
    """
    def _generate_data(n, f, mean, var, var2=1, d=2, gap=0):
        # Generate independent component of data
        # Assume isotropic Gaussian
        means = mean * np.ones(shape=(n, d-1))
        vrs = var * np.ones(shape=(n, d-1))
        x1 = np.random.normal(means, vrs, size=(n, d-1))
        eps = np.random.normal(0, var2, size=n)
        labels = np.sign(eps)
            
        #convert to 0,1
        labels = [int(p) if p==1 else 0 for p in labels]
        
        # Generate dependent component of data
        x2 = np.array(list(map(f, x1))) + eps + np.sign(eps)*gap
        x2 = np.expand_dims(x2, axis=1)
        # Merge x1 and x2
        features = np.concatenate([x1, x2], axis=1)
        return features, labels

    n = args.dataset.n
    m = args.dataset.m
    #f = lambda x : sum([np.sin(c) for c in x])
    f = lambda x : np.sin(sum([c for c in x]))
    id_mean = args.dataset.id_mean
    var = args.dataset.var
    d = args.model.in_features
    ood_mean = args.dataset.ood_mean
    gap = args.dataset.gap

    data_dict = {}
    
    data_dict['train'] = SyntheticDataset(*list(map(lambda x, dtype: torch.as_tensor(x, dtype=dtype), _generate_data(n, f, id_mean, var, d=d, gap=gap), [torch.float, torch.long])))
    data_dict['valid'] = SyntheticDataset(*list(map(lambda x, dtype: torch.as_tensor(x, dtype=dtype), _generate_data(m, f, id_mean, var, d=d, gap=gap), [torch.float, torch.long])))
    data_dict['d3m_train'] = data_dict['valid']
    data_dict['d3m_id'] = SyntheticDataset(*list(map(lambda x, dtype: torch.as_tensor(x, dtype=dtype), _generate_data(m, f, id_mean, var, d=d, gap=gap), [torch.float, torch.long])))
    data_dict['d3m_ood'] = SyntheticDataset(*list(map(lambda x, dtype: torch.as_tensor(x, dtype=dtype), _generate_data(m, f, ood_mean, var, d=d, gap=gap), [torch.float, torch.long])))

    return data_dict