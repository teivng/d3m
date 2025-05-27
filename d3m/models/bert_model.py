import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
import vbll
from .base import D3MAbstractModel
from ..configs import ModelConfig
from transformers import AutoTokenizer, AutoModel
from functools import partial

bert_dict = {
    'distilbert': "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
}


class BERTModel(D3MAbstractModel):
    """ D3M implementation with BERT features. """
    def __init__(self, cfg:ModelConfig, train_size:int):
        super(BERTModel, self).__init__()
        
        self.features = AutoModel.from_pretrained(bert_dict[cfg.bert_type])
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dict[cfg.bert_type])
        
        if cfg.freeze_features: 
            for param in self.features.parameters():
                param.requires_grad = False
    
        self.out_layer = vbll.DiscClassification(self.features.config.hidden_size, 
                                                 cfg.out_features, 
                                                 cfg.reg_weight_factor * 1/train_size, 
                                                 parameterization = cfg.param, 
                                                 return_ood=cfg.return_ood,
                                                 prior_scale=cfg.prior_scale, 
                                                 wishart_scale=cfg.wishart_scale
                                                 )
        self.cfg = cfg
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)  # call base method to move model
        self.device = next(self.features.parameters()).device  # update tracked device
        return self  # to allow chaining .to(...)
    
    def get_features(self, input_ids, attention_mask):
        x = self.features(input_ids, attention_mask).last_hidden_state[:,0,:]
        return x
    
    def forward(self, input_ids, attention_mask):
        x = self.get_features(input_ids, attention_mask)
        return self.out_layer(x)
  
    def get_last_layer(self):
        return self.out_layer
    
    def get_features_inference(self, x):
        tokenize = partial(
            self.tokenizer,
            padding='max_length',
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors='pt'
            
        )
        x = tokenize(x)
        output = self.get_features(x['input_ids'].to(self.device), \
            x['attention_mask'].to(self.device))
        return output
    
    def infer(self, x):
        output = self.get_features_inference(x)
        return self.out_layer(output)
    