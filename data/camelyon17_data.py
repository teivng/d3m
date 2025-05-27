import os 
import torch
from torchvision import transforms
from omegaconf import DictConfig
from torch.utils.data import Dataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .common import split_dataset


# ================================================
# ============Camelyon17 Utilities================
# ================================================


class InMemoryCamelyonSubset(Dataset):
    def __init__(self, wilds_subset, transform=None, device='cuda'):
        self.device = device
        self.transform = transform
        self._preload(wilds_subset)

    def _preload(self, wilds_subset):
        print(f"Preloading {len(wilds_subset)} samples into memory...")
        self.data = []
        for idx in tqdm(range(len(wilds_subset))):
            x, y, metadata = wilds_subset[idx]  # returns (image, label, metadata)
            if self.transform:
                x = self.transform(x)
            x = x.detach().to(self.device)
            y = torch.tensor(y).detach().to(self.device)
            self.data.append((x, y, metadata))
        print("Preloading complete.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, metadata = self.data[idx]
        return x, y, metadata


""" torchvision transforms """
camelyon17_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

camelyon17_val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_camelyon17_datasets(args:DictConfig):
    """Returns WILDS Camelyon17 Dataset objects
    We split the training set into a smaller validation set to train Phi.
    
    Returns:
        dict: contains the WILDS dataset splits already configured for D3M.
    """
    os.makedirs(args.dataset.data_dir, exist_ok=True)
    dataset = Camelyon17Dataset(root_dir=args.dataset.data_dir, download=args.dataset.download)
    
    dataset_dict = {}
    
    # Split the trainign set into train and id_valid
    ds_train = dataset.get_subset('train', frac=args.dataset.frac)
    ds_train, ds_id1_val = split_dataset(ds_train, lengths=[0.9, 0.1], random_seed=args.seed)
    
    ds_id2_val = dataset.get_subset('id_val', frac=args.dataset.frac)
    ds_ood = dataset.get_subset('test', frac=args.dataset.frac)
    
    # train d3m using the in-distribution validation set split from the training set
    dataset_dict = {
        'train': ds_train,
        'valid': ds_id1_val,
        'd3m_train': ds_id1_val,
        'd3m_id': ds_id2_val,
        'd3m_ood': ds_ood
    }
    
    for k in dataset_dict:
        if k == 'train':
            dataset_dict[k].dataset.transform = camelyon17_train_transform
        elif k in ['valid', 'd3m_train']:
            dataset_dict[k].dataset.transform = camelyon17_val_transform
        else:
            dataset_dict[k].transform = camelyon17_val_transform
    return dataset_dict
