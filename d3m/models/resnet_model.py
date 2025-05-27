import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
import vbll
from .base import D3MAbstractModel
from ..configs import ModelConfig


resnet_dict = {
    'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
    'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
    'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
    'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT, 2048),
}

class ResNetModel(D3MAbstractModel):
    """ D3M implementation with ResNet features. """
    def __init__(self, cfg:ModelConfig, train_size:int):
        super(ResNetModel, self).__init__()
        resnet_fn, resnet_weights, resnet_last_dim = resnet_dict[cfg.resnet_type]
        weights = resnet_weights if cfg.resnet_pretrained else None
        self.features = resnet_fn(weights=weights)
        if cfg.freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
                
        self.features.fc = nn.Linear(resnet_last_dim, cfg.hidden_dim)
        self.features.fc.requires_grad = True
        
        self.out_layer = vbll.DiscClassification(cfg.hidden_dim, 
                                                 cfg.out_features, 
                                                 cfg.reg_weight_factor * 1/train_size, 
                                                 parameterization = cfg.param, 
                                                 return_ood=cfg.return_ood,
                                                 prior_scale=cfg.prior_scale, 
                                                 wishart_scale=cfg.wishart_scale
                                                 )
        self.cfg = cfg
    
    def get_features(self, x):
        x = self.features(x)
        return x
    
    def forward(self, x):
        x = self.get_features(x)
        return self.out_layer(x)
  
    def get_last_layer(self):
        return self.out_layer
    