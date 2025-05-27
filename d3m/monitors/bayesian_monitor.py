from tqdm import tqdm
import wandb 
import numpy as np
import torch
import multiprocessing
from torch.utils.data import DataLoader, Dataset

from ..models.base import D3MAbstractModel
from ..configs import TrainConfig
from .utils import temperature_scaling, sample_from_dataset, get_class_from_string
from .monitor import D3MMonitor


class TempDataset(torch.utils.data.Dataset):
    ''' Temporary dataset to batch for data that does not fit on GPU '''
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  

 
class D3MBayesianMonitor(D3MMonitor):
    """Defines the Bayesian D3M Monitor (Algorithms 3 and 4).
    
    Attributes:
        model (D3MAbstractModel): the model class, i.e. hypothesis class for the base classifier
        trainset (Dataset): torch training dataset
        valset (Dataset): torch validation dataset
        train_cfg (TrainConfig): TrainConfig object configuring all aspects of training
        device (torch.device): torch.device, cuda or cpu
    """

    def __init__(self, *args, **kwargs):
        super(D3MBayesianMonitor, self).__init__(*args, **kwargs)
    

    def evaluate_model(self, loader, tqdm_enabled=False):
        running_loss = []
        running_acc = []
        with torch.no_grad():
            self.model.eval()
            for test_step, batch in enumerate(tqdm(loader, leave=False)):
                features, labels, *metadata = batch
                features, labels = features.to(self.device), labels.to(self.device)
                out = self.model(features)
                loss = out.val_loss_fn(labels)
                probs = out.predictive.probs
                acc = self.eval_acc(probs, labels).item()

                running_loss.append(loss.item())
                running_acc.append(acc)
                
        return running_loss, running_acc
    
    
    def train_model(self, tqdm_enabled=False, testloader=None):
        """Initial training of the model.

        Args:
            tqdm_enabled (bool, optional): Enables tqdm during training. Defaults to False.

        Returns:
            dict: dictionary containing training metrics
        """
        f = tqdm if tqdm_enabled else lambda x: x
        for epoch in f(range(self.train_cfg.num_epochs)):
            # =========================================================
            # ===================One Training Epoch====================
            # =========================================================        
                
            self.model.train()
            running_loss = []
            running_acc = []
            for train_step, batch in enumerate(tqdm(self.trainloader, leave=False)):
                features, labels, *_ = batch
                self.optimizer.zero_grad()
                features, labels = features.to(self.device), labels.to(self.device)
                out = self.model(features)
                loss = out.train_loss_fn(labels)
                probs = out.predictive.probs
                acc = self.eval_acc(probs, labels).item()
                running_loss.append(loss.item())
                running_acc.append(acc)
                loss.backward()
                self.optimizer.step()
                
            self.output_metrics['train_loss'].append(np.mean(running_loss))
            self.output_metrics['train_acc'].append(np.mean(running_acc))
            
            if self.verbose:
                print('Epoch: {:2d}, train loss: {:4.4f}\ttrain accuracy: {:4.4f}'.format(epoch, self.output_metrics['train_loss'][-1], self.output_metrics['train_acc'][-1]))
            # =========================================================
            # ==================Validate ID and OOD ===================
            # =========================================================
            if epoch % self.train_cfg.val_freq == 0:
                running_val_loss, running_val_acc = self.evaluate_model(self.valloader, tqdm_enabled=tqdm_enabled)
                self.output_metrics['val_loss'].append(np.mean(running_val_loss))
                self.output_metrics['val_acc'].append(np.mean(running_val_acc))
                if self.verbose:
                    print('Epoch: {:2d}, val loss: {:4.4f}\tval accuracy: {:4.4f}'.format(epoch, self.output_metrics['val_loss'][-1], self.output_metrics['val_acc'][-1]))
                if testloader:
                    running_test_loss, running_test_acc = self.evaluate_model(testloader, tqdm_enabled=tqdm_enabled)
                    self.output_metrics['ood_test_loss'].append(np.mean(running_test_loss))
                    self.output_metrics['ood_test_acc'].append(np.mean(running_test_acc))
                    if self.verbose:
                        print('Epoch: {:2d}, ood test loss: {:4.4f}\tood test accuracy: {:4.4f}'.format(epoch, self.output_metrics['ood_test_loss'][-1], self.output_metrics['ood_test_acc'][-1]))
            
            # wandb logging
            if wandb.run is not None:
                wandb.log({
                    'train_loss': self.output_metrics['train_loss'][-1],
                    'train_acc': self.output_metrics['train_acc'][-1],
                    'val_loss': self.output_metrics['val_loss'][-1],
                    'val_acc': self.output_metrics['val_acc'][-1],
                    'ood_test_loss': self.output_metrics['ood_test_loss'][-1] if testloader else None,
                    'ood_test_acc': self.output_metrics['ood_test_acc'][-1] if testloader else None,
                })

        return self.output_metrics
    
    
    def get_pseudolabels(self, X:torch.tensor):
        """Given samples X, return pseudolabels assigned by self.model

        Args:
            X (torch.tensor): input tensor

        Returns:
            torch.tensor: pseudolabels labeled by self.model
        """
        self.model.eval()
        
        '''loader = torch.utils.data.DataLoader(TempDataset(X, torch.arange(0, len(X))), batch_size=2048)
        y_hat_collection = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(self.device)
                features = self.model.get_features(X)
                ll_dist = self.model.out_layer.logit_predictive(features)
                y_hat = torch.argmax(ll_dist.loc, 1)
                y_hat_collection.append(y_hat)
        return torch.cat(y_hat_collection, dim=0)'''
        X = X.to(self.device)
        with torch.no_grad():
            features = self.model.get_features(X)
            ll_dist = self.model.out_layer.logit_predictive(features)
            y_hat = torch.argmax(ll_dist.loc, 1)
        return y_hat


    def compute_max_dis_rate(self, X:torch.tensor, y:torch.tensor, n_post_samples=5000, temperature=1):
        """Approximates the maximum disagreement rate among sampled weights.

        Args:
            X (torch.tensor): input tensor
            y (torch.tensor): pseudolabels
            n_post_samples (int, optional): number of posterior weights. Defaults to 5000.
            temperature (int, optional): softening of the logits. Defaults to 1.

        Returns:
            float: approximate maximum disagreement rate
        """
        self.model.eval()
        '''loader = torch.utils.data.DataLoader(TempDataset(X, y), batch_size=2048)
        maxes = [] 
        with torch.no_grad():
            for features, labels in loader:
                features, labels = features.to(self.device), labels.to(self.device)
                output = self.model.get_features(features)
                ll_dist = self.model.out_layer.logit_predictive(output)
                logits_samples = ll_dist.rsample(sample_shape=torch.Size([n_post_samples]))
            
                # scale down with temperature
                logits_samples = temperature_scaling(logits_samples, temperature)
                y_hat = torch.argmax(logits_samples, -1)
                y_tile = torch.tile(labels, (n_post_samples, 1)).cuda()
                dis_mat = (y_hat != y_tile)
                dis_rate = dis_mat.sum(dim=-1)/len(y)
                maxes.append(torch.max(dis_rate).item())
        return max(maxes)'''
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            output = self.model.get_features(X)
            ll_dist = self.model.out_layer.logit_predictive(output)
            logits_samples = ll_dist.rsample(sample_shape=torch.Size([n_post_samples]))
            logits_samples = temperature_scaling(logits_samples, temperature)
            #y_hat = torch.argmax(logits_samples, -1)
            dist = torch.distributions.Categorical(logits=logits_samples)
            y_hat = dist.sample()
            y_tile = torch.tile(y, (n_post_samples, 1)).to(self.device)
            dis_mat = (y_hat != y_tile)
            dis_rate = dis_mat.sum(dim=-1)/len(y)
        return torch.max(dis_rate).item()