import os 
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from omegaconf import DictConfig

from .common import TensorDataset

# ================================================
# =============CIFAR-10 Utilities=================
# ================================================


""" torchvision transforms """
cifar10_train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomCrop(size=[32,32], padding=4),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
])
cifar10_test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
])

    
class CIFAR101Dataset(TensorDataset):
    """CIFAR10.1 Dataset class

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
    """
    
    def __init__(self, X, y):
        super(CIFAR101Dataset, self).__init__(X=X, y=y)
        
    def __str__(self):
        return """Dataset CIFAR10.1
    \tNumber of datapoints: {}
    \tRoot location: data/cifar10_data/
    \tSplit: OOD
    \tTestTransform
    \t{}
    """.format(self.__len__(), cifar10_test_transforms)


def get_cifar10_datasets(args:DictConfig, download=True):
    """Returns processed CIFAR10 and CIFAR10.1 Dataset objects
    We split the train dataset into a trainset and a validation set
    The validation set has two purposes: 
        - Validate the training of the base model
        - Train the distribution of in-distribution maximum disagreement rates (Phi)
    The CIFAR-10 test set will be used to validate the training of Phi, i.e. an iid_test sample
    The CIFAR-10.1 o.o.d. set will be used to study the TPR of D3M.

    Returns:
        tuple(Dataset, Dataset, Dataset, CIFAR101Dataset): 4-tuple containing:
        CIFAR10 train, test, train with test transforms, and CIFAR10.1. 
    """
    os.makedirs(args.dataset.data_dir, exist_ok=True)
    # Loads the cifar-10 test set
    cifar10test = torchvision.datasets.CIFAR10(root=args.dataset.data_dir, 
                                               train=False, 
                                               transform=cifar10_test_transforms, 
                                               download=download)
    
    # make the cifar-10 train and validation sets
    cifar10train = torchvision.datasets.CIFAR10(root=args.dataset.data_dir,
                                                train=True, 
                                                transform=None,
                                                download=download)
    
    cifar10train, cifar10val = torch.utils.data.random_split(cifar10train, [40000, 10000])
    cifar10train = torch.utils.data.Subset(
        dataset=torchvision.datasets.CIFAR10(
            root=args.dataset.data_dir,
            train=True,
            transform=cifar10_train_transforms,
            download=True
        ),
        indices=cifar10train.indices
    )
    cifar10val = torch.utils.data.Subset(
        dataset=torchvision.datasets.CIFAR10(
            root=args.dataset.data_dir,
            train=True,
            transform=cifar10_test_transforms,
            download=True
        ),
        indices=cifar10val.indices
    )
    
    # Ensure CIFAR-10.1 data is in "data/" directory
    with open(os.path.join(args.dataset.data_dir, 'cifar10.1_v6_data.npy'), 'rb') as f:
        ood_data = np.load(f)
    with open(os.path.join(args.dataset.data_dir, 'cifar10.1_v6_labels.npy'), 'rb') as f:
        ood_labels = np.load(f)
    
    transformed101data = torch.zeros(size=(len(ood_data), 3, 32, 32))
    for idx in range(len(ood_data)):
        transformed101data[idx] = cifar10_test_transforms(ood_data[idx])
    transformed101labels = torch.as_tensor(ood_labels, dtype=torch.long)
    cifar101 = CIFAR101Dataset(transformed101data, transformed101labels)
    return cifar10train, cifar10val, cifar10test, cifar101
