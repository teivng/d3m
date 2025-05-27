from dataclasses import dataclass
from pprint import pprint

@dataclass
class ModelConfig:
    # standard stuff
    name: str
    out_features: int
    
    # vbll-specific configs
    reg_weight_factor: float
    param: str
    prior_scale: float
    wishart_scale: float
    
    def __str__(self):
        self.print_config()
        return ''
    def print_config(self):
        pprint(vars(self))
    
    
@dataclass
class ConvModelConfig(ModelConfig):
    """Configuration for the base CNN model"""
    in_channels: int
    mid_channels: int
    kernel_size: int
    mid_layers: int
    pool_dims: int
    hidden_dim: int
    dropout: float
    return_ood: bool = False
    
    
@dataclass
class MLPModelConfig(ModelConfig):
    """Configuration for the base MLP model"""
    in_features: int
    mid_features: int
    mid_layers: int
    dropout: float
    return_ood: bool = False
    
    
@dataclass 
class ResNetModelConfig(ModelConfig):
    """Configuration for ResNet models"""
    resnet_type: str
    hidden_dim: int
    resnet_pretrained: bool = False
    freeze_features: bool = False
    return_ood: bool = False
    
    
@dataclass 
class BERTModelConfig(ModelConfig):
    """Configuration for BERT-based models"""
    bert_type: str
    max_length: int
    freeze_features: bool = False
    return_ood: bool = False