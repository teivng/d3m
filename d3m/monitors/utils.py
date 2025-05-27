from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from torch.utils.data import DataLoader, Dataset

class MaskedDataset(Dataset):
    def __init__(self, dataset: Dataset, mask=True):
        self.dataset = dataset
        self.mask = torch.tensor(mask, dtype=torch.bool)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, self.mask


def temperature_scaling(logits, temperature):
    return logits / temperature


def sample_from_dataset(n_samples:int, dataset:Dataset, replace=True):
    """Given a dataset, sample n_samples.

    Args:
        n_samples (int): number of samples to sample
        dataset (Dataset): torch Dataset object
        replace (bool, optional): Whether to sample with replacement. Defaults to True.

    Returns:
        torch.tensor: sample from dataset.
        torch.tensor: true labels of the samples
    """
    data_size = dataset[0][0].size()
    indices = np.random.choice(np.arange(len(dataset)), n_samples, replace=replace)
    tmp = torch.zeros(size=(n_samples, *data_size))
    true_labels = torch.zeros(size=(n_samples,))
    for i in range(len(indices)):
        tmp[i] = dataset[indices[i]][0]
        true_labels[i] = dataset[indices[i]][1]
    return tmp, true_labels

def joint_sample_from_datasets(n_samples: int, datasetA:Dataset, datasetB: Dataset, balance_ratio:float = 1, replace=True):
    """
    Given two datasets, sample (n * balance_ratio) samples from A, and (n * (1-balance_ratio)) samples from B
    Args:
        n_samples (int): number of samples to sample
        datasetA (Dataset): torch Dataset object
        datasetB (Dataset): torch Dataset object
        balance_ratio (float): ratio of samples to sample from datasetA
        replace (bool, optional): Whether to sample with replacement. Defaults to True.

    Returns:
        torch.tensor: sample from dataset.
        torch.tensor: mask of the samples, 1 if sample is from datasetA, 0 if sample is from datasetB
    """
    
    assert 0 <= balance_ratio <= 1, "balance_ratio must be between 0 and 1"
    n_samples_A = int(n_samples * balance_ratio)
    n_samples_B = n_samples - n_samples_A
    data_size = datasetA[0][0].size()
    indices_A = np.random.choice(np.arange(len(datasetA)), n_samples_A, replace=replace)
    indices_B = np.random.choice(np.arange(len(datasetB)), n_samples_B, replace=replace)
    tmp = torch.zeros(size=(n_samples, *data_size))
    true_labels = torch.zeros(size=(n_samples,))
    mask = torch.zeros(size=(n_samples,))
    for i in range(len(indices_A)):
        tmp[i] = datasetA[indices_A[i]][0]
        true_labels[i] = datasetA[indices_A[i]][1]
        mask[i] = 1
    for i in range(len(indices_B)):
        tmp[i + n_samples_A] = datasetB[indices_B[i]][0]
        true_labels[i + n_samples_A] = datasetB[indices_B[i]][1]
    return tmp, true_labels, mask


def get_class_from_string(class_string):
    """Given a class name, return the module to be instantiated.
    To be used to retrieve objects like "torch.optim.AdamW". 

    Args:
        class_string (str): string name of the package to be retrieved

    Returns:
        <class 'module'>: the class to be retrieved.
    """
    # Split the string into module and class name
    module_name, class_name = class_string.rsplit('.', 1)
    
    # Import the module dynamically
    module = __import__(module_name, fromlist=[class_name])
    
    # Get the class from the module
    cls = getattr(module, class_name)
    
    return cls


class FILoss(torch.nn.Module):
    def __init__(self, weight=None, alpha=None):
        super(FILoss, self).__init__()
        self.weight = weight
        self.alpha = alpha

    def forward(self, logits, labels, mask):
        return self.full_info_loss(logits, labels,
                        mask, alpha=self.alpha, weight=self.weight)
    
    def full_info_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, alpha: Optional[float] = None,
             weight=None) -> torch.Tensor:
        """
        :param logits: (batch_size, num_classes) tensor of logits
        :param labels: (batch_size,) tensor of labels
        :param mask: (batch_size,) mask
        :param alpha: (float) weight of datasetB samples
        :param weight:  (torch.Tensor) weight for each sample_data, default=None do not apply weighting
        :return: (tensor, float) the disagreement cross entropy loss
        """
        alpha=self.alpha
        

        if mask.all():
            # if all labels are positive, then use the standard cross entropy loss
            return F.cross_entropy(logits, labels)
        
        if alpha is None:
            alpha = 1 / (1 + (~mask).float().sum())
            # 1 / (1 + #negative samples (rejection))

        num_classes = logits.shape[1]

        q_logits, q_labels = logits[~mask], labels[~mask]


        zero_hot = 1. - F.one_hot(q_labels, num_classes=num_classes)
        ce_n = -(q_logits * zero_hot).sum(dim=1) / (num_classes - 1) + torch.logsumexp(q_logits, dim=1)

        if torch.isinf(ce_n).any() or torch.isnan(ce_n).any():
            raise RuntimeError('NaN or Infinite loss encountered for ce-q')

        if (~mask).all():
            return (ce_n * alpha).mean()

        p_logits, p_labels = logits[mask], labels[mask]

        ce_p = F.cross_entropy(p_logits, p_labels, reduction='none', weight=weight)
        return torch.cat([ce_n * alpha, ce_p]).mean()