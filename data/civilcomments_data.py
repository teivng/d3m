import os
import torch
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
from d3m.models.bert_model import bert_dict
from tqdm.auto import tqdm
from .common import split_dataset, TensorDataset
from functools import partial
from typing import Union 


# ================================================
# ===========CivilComments Utilities==============
# ================================================


class FeaturizedBERTDataset(Dataset):
    """Featurized BERT dataset class..

    Attributes:
        X (torch.tensor): dataset features
        y (torch.tensor): dataset labels
        metadata (torch.tensor): metadata
    """
    def __init__(self, X, y, metadata):
        super(FeaturizedBERTDataset, self).__init__()
        self.X = X
        self.y = y
        self.metadata = metadata
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.metadata[idx]


class TokenizedBERTDataset(Dataset):
    """Tokenized BERT dataset class.

    Attributes:
        input_ids (torch.tensor): tokenized input
        attention_mask (torch.tensor): attention mask
        y (torch.tensor): dataset labels
        metadata (torch.tensor): metadata
    """
    def __init__(self, input_ids, attention_mask, y, metadata):
        super(TokenizedBERTDataset, self).__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.y = y
        self.metadata = metadata
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx]), self.y[idx], self.metadata[idx]

    def sample(self, n: int, replace: bool = True):
        """Samples n elements from the dataset.

        Args:
            n (int): number of samples to take
            replace (bool): whether to sample with replacement

        Returns:
            TokenizedBERTDataset: sampled dataset
        """
        if not replace and n > len(self):
            raise ValueError(f"Cannot sample {n} elements without replacement from dataset of size {len(self)}.")

        indices = torch.randint(0, len(self), (n,)) if replace else torch.randperm(len(self))[:n]

        return TokenizedBERTDataset(
            self.input_ids[indices],
            self.attention_mask[indices],
            self.y[indices],
            self.metadata[indices]
    )
        
    def to(self, device:Union[str, torch.device]):
        """Moves the dataset to the specified device.
        Keeps metadata on CPU.
        Args:
            device (Union[str, torch.device]): device to move the dataset to
        """
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.y = self.y.to(device)
        return self
    
def get_civilcomments_datasets_tokenized(args:DictConfig):
    """Returns WILDS CivilComments Dataset objects TOKENIZED
    We split the training set into TWO smaller validation sets set to train Phi and perform id-validation.
    
    Returns:
        dict: contains the WILDS dataset splits already configured for D3M.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    processed_dataset_path = f'data/civilcomments_v1.0/{args.model.bert_type}_tokenized.pt'
    if os.path.exists(processed_dataset_path):
        print('BERT-preprocessed CivilComments dataset exists! Loading...')
        tokenized_datasets = torch.load(processed_dataset_path, weights_only=False)
        print('Loading succesful!')
        
        # fake dsets
        dsets = {k: [] for k in ['train', 'id1', 'id2', 'test']}
    else:
        print('BERT-preprocessed CivilComments dataset not found. Processing...')
        os.makedirs(args.dataset.data_dir, exist_ok=True)
        dataset = CivilCommentsDataset(root_dir=args.dataset.data_dir, download=args.dataset.download)
        
        trainset = dataset.get_subset('train', frac=args.dataset.frac)
        testset = dataset.get_subset('test', frac=args.dataset.frac)
        
        # Split training 80-10-10
        new_trainset, id_val1, id_val2 = split_dataset(
            trainset, 
            [0.8, 0.1, 0.1],
            random_seed=args.seed
        )
        
        # Pre-compute BERT features
        dsets = {
            'train': new_trainset,
            'id1': id_val1,
            'id2': id_val2,
            'test': testset
        }
        
        tokenizer = AutoTokenizer.from_pretrained(bert_dict[args.model.bert_type])
        
        tokenize = partial(
            tokenizer,
            padding='max_length',
            truncation=True, 
            max_length=args.model.max_length,
            return_tensors='pt'
        )
        
        input_ids_dict = {k: [] for k in dsets.keys()}
        attention_mask_dict = {k: [] for k in dsets.keys()}    
        labels = {k: [] for k in dsets.keys()}
        metadata = {k: [] for k in dsets.keys()}
        
        # tokenize everything
        for k in dsets.keys():
            loader = DataLoader(dsets[k], batch_size=512, num_workers=args.train.num_workers)
            print(f'tokenizing {k} split...')
            for i, (text, toxic, meta) in enumerate(tqdm(loader)):
                with torch.no_grad():
                    try:
                        tokenized = tokenize(text)
                        input_ids_dict[k].append(tokenized['input_ids'])
                        attention_mask_dict[k].append(tokenized['attention_mask'])
                        labels[k].append(toxic)
                        metadata[k].append(meta)
                    except Exception as e:  
                        print(f'Exception found in split {k}, batch {i}. Skipping...')
                        continue
        # concatenate 
        input_ids, attention_masks, labels, metadata = (
            {k: torch.cat(v, dim=0) for k, v in d.items()}
            for d in (input_ids_dict, attention_mask_dict, labels, metadata)
        )
        torch.save({
            'input_ids': input_ids_dict,
            'attention_mask': attention_mask_dict,
            'labels': labels,
            'metadata': metadata,
        }, processed_dataset_path)
    
    tmp = {k: TokenizedBERTDataset(
        tokenized_datasets['input_ids'][k],
        tokenized_datasets['attention_masks'][k], # @todo change this when tokenize again
        tokenized_datasets['labels'][k],
        tokenized_datasets['metadata'][k]
    ) for k in dsets.keys()}
    
    dataset_dict = {
        'train': tmp['train'],
        'valid': tmp['id1'],
        'd3m_train': tmp['id1'],
        'd3m_id': tmp['id2'],
        'd3m_ood': tmp['test']
    } 
    
    '''Load Jigsaw dataset'''
    jigsaw_dataset_path = f'data/jigsaw-toxic-comments/{args.model.bert_type}_tokenized.pt'
    if os.path.exists(jigsaw_dataset_path):
        print('BERT-preprocessed Jigsaw dataset exists! Loading...')
        jigsaw_dataset = torch.load(jigsaw_dataset_path, weights_only=False)
        print('Loading succesful!')
        jigsaw = TokenizedBERTDataset(
            jigsaw_dataset['input_ids'],
            jigsaw_dataset['attention_mask'],
            jigsaw_dataset['labels'],
            jigsaw_dataset['metadata']
        )
    else:
        print('BERT-preprocessed Jigsaw dataset not found. Processing...')
    
    dataset_dict['d3m_ood'] = jigsaw
    return dataset_dict


def get_civilcomments_datasets_featurized(args:DictConfig):
    """Returns WILDS CivilComments Dataset objects FEATURIZED
    We split the training set into TWO smaller validation sets set to train Phi and perform id-validation.
    
    Returns:
        dict: contains the WILDS dataset splits already configured for D3M.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    processed_dataset_path = f'data/civilcomments_v1.0/{args.model.bert_type}_featurized.pt'
    if os.path.exists(processed_dataset_path):
        print('BERT-preprocessed dataset exists! Loading...')
        featurized_datasets = torch.load(processed_dataset_path, weights_only=False)
        print('Loading succesful!')
        
        # fake dsets
        dsets = {k: [] for k in ['train', 'id1', 'id2', 'test']}
    else:
        print('BERT-preprocessed dataset not found. Processing...')
        os.makedirs(args.dataset.data_dir, exist_ok=True)
        dataset = CivilCommentsDataset(root_dir=args.dataset.data_dir, download=args.dataset.download)
        
        trainset = dataset.get_subset('train', frac=args.dataset.frac)
        testset = dataset.get_subset('test', frac=args.dataset.frac)
        
        # Split training 80-10-10
        new_trainset, id_val1, id_val2 = split_dataset(
            trainset, 
            [0.8, 0.1, 0.1],
            random_seed=args.seed
        )
        
        # Pre-compute BERT features
        dsets = {
            'train': new_trainset,
            'id1': id_val1,
            'id2': id_val2,
            'test': testset
        }
        
        tokenizer = AutoTokenizer.from_pretrained(bert_dict[args.model.bert_type])
        bert = AutoModel.from_pretrained(bert_dict[args.model.bert_type]).to(device)

        tokenize = partial(
            tokenizer,
            padding='max_length',
            truncation=True, 
            max_length=args.model.max_length,
            return_tensors='pt'
        )
        
        features = {k: [] for k in dsets.keys()}
        labels = {k: [] for k in dsets.keys()}
        metadata = {k: [] for k in dsets.keys()}
        
        # tokenize everything
        for k in dsets.keys():
            loader = DataLoader(dsets[k], batch_size=512, num_workers=args.train.num_workers)
            print(f'featurizing {k} split...')
            for i, (text, toxic, meta) in enumerate(tqdm(loader)):
                with torch.no_grad():
                    try:
                        tokens = tokenize(text).to(device)
                        output = bert(**tokens).last_hidden_state[:, 0, :]
                        output = output.cpu()
                        features[k].append(output)
                        labels[k].append(toxic)
                        metadata[k].append(meta)
                    except:
                        print(f'Exception found in split {k}, batch {i}. Skipping...')
                        continue
        
        # concatenate 
        features, labels, metadata = (
            {k: torch.cat(v, dim=0) for k, v in d.items()}
            for d in (features, labels, metadata)
        )
        
        # save
        featurized_datasets = {
            'features': features,
            'labels': labels,
            'metadata': metadata
        }
        torch.save(featurized_datasets, processed_dataset_path)
    
    tmp = {k: FeaturizedBERTDataset(
        featurized_datasets['features'][k],
        featurized_datasets['labels'][k],
        featurized_datasets['metadata'][k]
        
    ) for k in dsets.keys()}
    
    dataset_dict = {
        'train': tmp['train'],
        'valid': tmp['id1'],
        'd3m_train': tmp['id1'],
        'd3m_id': tmp['id2'],
        'd3m_ood': tmp['test']
    } 
    
    return dataset_dict
    
