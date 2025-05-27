import torch.nn as nn
import torch.nn.functional as F
import vbll
from .base import D3MAbstractModel
from ..configs import ModelConfig


class ConvModel(D3MAbstractModel):
    """D3M implementation with CNN features."""
    
    def __init__(self, cfg:ModelConfig, train_size:int):
        assert train_size is not None
        super(ConvModel, self).__init__()
        init_kernel_size = cfg.kernel_size + 2
        self.init_conv = nn.Conv2d(cfg.in_channels, cfg.mid_channels, kernel_size=init_kernel_size, padding=int((init_kernel_size-1)/2))
        
        self.mid_convs = nn.ModuleList(
            [nn.Conv2d(cfg.mid_channels, cfg.mid_channels, kernel_size=cfg.kernel_size, padding=int((cfg.kernel_size-1)/2)) for _ in range(cfg.mid_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(cfg.mid_channels) for _ in range(cfg.mid_layers)]
        )
        flatten_dim = int(cfg.mid_channels * (32/4) ** 2)
        self.fc = nn.Linear(flatten_dim, cfg.hidden_dim)
        self.out_layer = vbll.DiscClassification(cfg.hidden_dim, 
                                                 cfg.out_features, 
                                                 cfg.reg_weight_factor * 1/train_size, 
                                                 parameterization = cfg.param, 
                                                 return_ood=cfg.return_ood,
                                                 prior_scale=cfg.prior_scale, 
                                                 wishart_scale=cfg.wishart_scale
                                                 )
                                            
        self.pool = nn.MaxPool2d(cfg.pool_dims, cfg.pool_dims)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.cfg = cfg
    
    def get_features(self, x):
        # initial convolution
        x = F.elu(self.init_conv(x))
        x = self.dropout(self.pool(x))
        # mid convolutions with skip connections
        for idx in range(len(self.mid_convs)):
            identity = x
            out = self.mid_convs[idx](x)
            out = self.bns[idx](out)
            out += identity
            x = self.dropout(F.elu(out))
        
        x = self.pool(x).view(x.size()[0], -1)
    
        x = self.dropout(F.elu(self.fc(x)))
        return x
    
    def forward(self, x):
        x = self.get_features(x)
        return self.out_layer(x)
  
    def get_last_layer(self):
        return self.out_layer