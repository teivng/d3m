import torch.nn as nn
import torch.nn.functional as F

import vbll
from .base import D3MAbstractModel
from ..configs import ModelConfig


class MLPModel(D3MAbstractModel):
    """D3M implementation with tabular features."""
    def __init__(self, cfg:ModelConfig, train_size:int):
        assert train_size is not None
        super(MLPModel, self).__init__()

        self.init_fc = nn.Linear(cfg.in_features, cfg.mid_features)
        
        self.mid_fc = nn.ModuleList([
            nn.Linear(cfg.mid_features, cfg.mid_features) for _ in range(cfg.mid_layers)
        ])
        
        self.out_layer = vbll.DiscClassification(cfg.mid_features, 
                                                 cfg.out_features, 
                                                 cfg.reg_weight_factor * 1/train_size, 
                                                 parameterization = cfg.param, 
                                                 return_ood=cfg.return_ood,
                                                 prior_scale=cfg.prior_scale, 
                                                 wishart_scale=cfg.wishart_scale
                                                 )
        
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.cfg = cfg
    
    def get_features(self, x):
        x = self.dropout(F.elu(self.init_fc(x)))
        for f in self.mid_fc:
            identity = x
            out = f(x)
            out += identity
            x = self.dropout(F.elu(out))
        
        return x
    
    def forward(self, x):
        x = self.get_features(x)
        return self.out_layer(x)

    def get_last_layer(self):
        return self.out_layer
        
    