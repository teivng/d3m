import torch
import numpy as np
from torchvision.transforms import v2 
import torchvision
import inspect
from d3m.configs import TrainConfig
from d3m.configs import ConvModelConfig, MLPModelConfig, ResNetModelConfig, BERTModelConfig
from data import get_cifar10_datasets, get_uci_datasets, get_synthetic_datasets, get_camelyon17_datasets, get_civilcomments_datasets_featurized, get_civilcomments_datasets_tokenized
from omegaconf import DictConfig, OmegaConf


def print_args_and_kwargs(*args, **kwargs):
    """Prints all args and kwargs"""
    # Print all positional arguments (*args)
    print("Positional arguments (*args):")
    for i, arg in enumerate(args, start=1):
        print(f"  Argument {i}: {arg}")

    # Print all keyword arguments (**kwargs)
    print("\nKeyword arguments (**kwargs):")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    

def filter_args(cls, args):
    """Given parsed arguments, filter arguments required by the dataclass

    Args:
        cls: class to get the constructor signature from
        args (argparse.Namespace): parsed arguments

    Returns:
        dict: dictionary with a kv pair for each constructor parameter and the matched argument parsed. 
    """
    
    sig = inspect.signature(cls.__init__)  # Get constructor signature
    return {k: v for k, v in vars(args).items() if k in sig.parameters}


def get_configs(args:DictConfig):
    """From parsed arguments, generate ModelConfig and TrainConfig configs for the experiment.

    Args:
        args (argparse.Namespace): hydra argument

    Returns:
        tuple(ModelConfig, TrainConfig): 2-tuple containing:
        the model and train configs respectively.
    """
    Configs = {
        'cifar10': ConvModelConfig,
        'uci': MLPModelConfig,
        'synthetic': MLPModelConfig,
        'camelyon17': ResNetModelConfig,
        'civilcomments': BERTModelConfig,
    }
    model_args = OmegaConf.to_container(args.model)
    train_args = OmegaConf.to_container(args.train)
    model_args['out_features'] = args.dataset.num_classes
    model_config = Configs[args.dataset.name](**model_args)
    train_config = TrainConfig(**train_args)
    
    return model_config, train_config


def get_datasets(args:DictConfig):
    """Generic class to get datasets. 
    
    Args:
        args (omegaconf.DictConfig): hydra config

    Returns:
        dict: dictionary with keys:
                - train         (used to train base model)
                - valid         (used ot validate base model)
                - d3m_train   (used to train d3m's Phi)
                - d3m_id      (used to validate FPR)
                #- d3m_ood     (used to validate TPR)
    """
    dataset_dict = {}
    name = args.dataset.name
    
    if name == 'cifar10':
        train, val, id, ood = get_cifar10_datasets(args, download=True)
        dataset_dict['train'] = train
        dataset_dict['valid'] = val
        dataset_dict['d3m_train'] = val         # train d3m using the CIFAR-10 validation set (10k)
        dataset_dict['d3m_id'] = id             # validate d3m using the CIFAR-10 test set (10k)
        dataset_dict['d3m_ood'] = ood
    
    elif name == 'uci':
        uci_dict = get_uci_datasets(args)
        dataset_dict['train'] = uci_dict['train']
        dataset_dict['valid'] = uci_dict['val']
        dataset_dict['d3m_train'] = uci_dict['val']
        dataset_dict['d3m_id'] = uci_dict['iid_test']
        dataset_dict['d3m_ood'] = uci_dict['ood_test']
    
    elif name == 'synthetic':
        dataset_dict = get_synthetic_datasets(args)
    
    elif name == 'camelyon17':
        dataset_dict = get_camelyon17_datasets(args)
    elif name == 'civilcomments':
        if args.dataset.featurized:
            print('Featurized dataset requested. Using featurized pipeline.')
            dataset_dict = get_civilcomments_datasets_featurized(args)
        else:
            print('Tokenized dataset requested. Using tokenized pipeline.')
            dataset_dict = get_civilcomments_datasets_tokenized(args)
        
    else: 
        raise NameError('Not a dataset with a known data pipeline implementation.')
    return dataset_dict