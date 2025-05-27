from abc import ABC, abstractmethod
from ..models.base import D3MAbstractModel
from .utils import get_class_from_string, sample_from_dataset
from ..configs import TrainConfig

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple, List

class D3MMonitor(ABC):
    """Defines the abstract D3M Monitor class. All monitors should inherit from this class.
    
    Attributes:
        model (D3MAbstractModel): the model class, i.e. hypothesis class for the base classifier
        trainset (Dataset): torch training dataset
        valset (Dataset): torch validation dataset
        train_cfg (TrainConfig): TrainConfig object configuring all aspects of training
        device (torch.device): torch.device, cuda or cpu
        
    Methods:
        - Implemented:
            - eval_acc: evaluates the accuracy of the model
            - d3m_test: runs one round of the d3m_test on a subsample of 
                          the given dataset. Used both to generate the distribution
                          of in-distribution disagreement rates self.Phi and 
                          to run the test (Deploy). 
            - pretrain_disagreement_distribution: generates self.Phi
            - repeat_tests: repeats the D3M Deploy stage (Deploy) on the given dataset
                            useful in computing FPRs and TPRs. 
        - Need to override:
            - train_model: trains the base classifier self.model
            - get_pseudolabels: given some data, assign pseudolabels according to self.model
            - compute_max_dis_rate: given data and pseudolabels, computes the maximum disagreement
                                    rate achievable by models from the same hypothesis class
                                    as self.models
    """
    def __init__(self, 
                 model:D3MAbstractModel,
                 trainset: Dataset,
                 valset: Dataset,
                 train_cfg: TrainConfig,
                 device=torch.device,
                 verbose=True):
        
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.train_cfg = train_cfg
        self.device = device
        self.verbose = verbose
        
        # Get optimizer from string
        opt_cls = get_class_from_string(train_cfg.optimizer)
        self.optimizer = opt_cls(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.wd,
        )
        
        self.trainloader = DataLoader(self.trainset, 
                                      batch_size=train_cfg.batch_size, 
                                      shuffle=True, 
                                      num_workers=train_cfg.num_workers, 
                                      pin_memory=train_cfg.pin_memory)
        self.valloader = DataLoader(self.valset, 
                                    batch_size=train_cfg.batch_size, 
                                    shuffle=True, 
                                    num_workers=train_cfg.num_workers, 
                                    pin_memory=train_cfg.pin_memory)

        self.output_metrics = {
            'train_loss': [],
            'val_loss': [],
            'ood_test_loss': [],
            'train_acc': [],
            'val_acc': [],
            'ood_test_acc': [],
            'ood_auroc': []
        }
        
        # For D3M pretraining
        self.Phi = []
        self.replace = True
        # To device
        self.model = self.model.to(self.device)
        
    
    def eval_acc(self, preds:torch.tensor, y:torch.tensor) -> float:
        """Evaluates the accuracy of the model.

        Args:
            preds (torch.tensor): predictions
            y (torch.tensor): labels

        Returns:
            float: accuracy score of prediction
        """
        map_preds = torch.argmax(preds, dim=1)
        return (map_preds == y).float().mean()
    
    
    def d3m_test(self, dataset:Dataset, data_sample_size:int=1000, alpha=0.95, replace=True, *args, **kwargs) -> Tuple[float, bool]:
        """Given a dataset, computes the maximum disagreement rate as well as the OOD verdict.
        Used to both generate Phi and Algorithms 2 and 4.

        Args:
            dataset (Dataset): dataset object
            data_sample_size (int, optional): size of bootstraped dataset. Defaults to 1000
            alpha (float): statistical power of the test. Defaults to 0.95
            replace (bool): sample with replacement. Defaults to True

        Returns:
            tuple(float, bool): 2-tuple containing:
            - maximum disagreement rate achievable by models from the same hypothesis class
                while (approximately) maintaining correctness on training set
            - OOD verdict w.r.t. self.Phi
        """
        with torch.no_grad():
            X, _ = sample_from_dataset(n_samples=data_sample_size, dataset=dataset, replace=replace)
            y_pseudo = self.get_pseudolabels(X)
            max_dis_rate = self.compute_max_dis_rate(X, y_pseudo, *args, **kwargs)


        return max_dis_rate, max_dis_rate >= np.quantile(self.Phi, alpha) if self.Phi != [] else 0 
    
    
    def pretrain_disagreement_distribution(self, dataset:Dataset, Phi_size=500, tqdm_enabled=True, *args, **kwargs) -> None:
        """Given a dataset, generates the Phi distribution of maximum disagreement rates.
        Uses self.d3m_test to compute Phi. 

        Args:
            dataset (Dataset): dataset object, should be in-distribution
            Phi_size (int, optional): size of phi. Defaults to 500.
        """
        f = tqdm if tqdm_enabled else lambda x: x 
        self.model.eval()
        with torch.no_grad():
            for i in f(range(Phi_size)):
                max_dis_rate, _ = self.d3m_test(dataset, replace=self.replace, *args, **kwargs)
                self.Phi.append(max_dis_rate)
    
    
    def repeat_tests(self, dataset:Dataset, n_repeats=1, *args, **kwargs) -> Tuple[float, List] :
        """After training Phi, monitors D-PDD using D3M Deploy on dataset.
    
        Args:
            dataset (Dataset): dataset to run D3M Deploy
            n_repeats (int, optional): number of times to repeat independent realizations 
                of D3M Deploy (TPR/FPR calculations). Defaults to 1.
        """
        assert self.Phi != []
        self.model.eval()
        with torch.no_grad():
            tprs = []
            max_dis_rates = []
            for i in tqdm(range(n_repeats)):
                max_dis_rate, result = self.d3m_test(dataset=dataset, replace=self.replace, *args, **kwargs)
                tprs.append(result)
                max_dis_rates.append(max_dis_rate)
            return np.mean(tprs), max_dis_rates
    
    
    @abstractmethod
    def train_model(self, *args, **kwargs) -> dict:
        """Abstract training function
        
        Trains self.model on self.trainloader, validates self.model on 
        self.testloader.
        
        Returns:
            dict: dictionary containing training metrics. 
        """
        pass
    
    @abstractmethod
    def evaluate_model(self, loader:DataLoader, tqdm_enabled=False) -> Tuple[float, float]:
        """Abstract evaluation function
        
        Evaluates self.model on the given loader. 

        Args:
            loader (DataLoader): dataloader to evaluate on
            tqdm_enabled (bool, optional): enables tqdm during evaluation. Defaults to False.

        Returns:
            tuple(list, list): 2-tuple containing:
                - running loss
                - running accuracy
        """
        pass
    
    @abstractmethod
    def get_pseudolabels(self, X:torch.tensor, *args, **kwargs) -> torch.tensor:
        """Given a set of data points, assign pesudolabels according to the base model self.model

        Args:
            X (torch.tensor): input tensor to be labeled
            
        Returns:
            torch.tensor: pseudolabels according to self.model
        """
        pass
    
    
    @abstractmethod
    def compute_max_dis_rate(self, X:torch.tensor, y:torch.tensor, *args, **kwargs) -> float:
        """Given data points and their PSEUDOlabels, compute the maximum disagreement rate
        achievable by models in the same hypothesis class as self.model

        Args:
            X (torch.tensor): input tensor
            y (torch.tensor): pseudolabels

        Returns:
            float: maximum disagreeemnt rate achievable etc...
        """
        pass