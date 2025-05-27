from tqdm.auto import tqdm
import torch
import wandb
import numpy as np


from .bayesian_monitor import D3MBayesianMonitor
from .utils import temperature_scaling


from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from torch.utils.data import Dataset

from typing import Tuple
from torcheval.metrics import BinaryF1Score


def make_bert_input(input_ids, attention_mask, device, to_device=False):
    input_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    if to_device:
        input_dict['input_ids'] = input_dict['input_ids'].to(device)
        input_dict['attention_mask'] = input_dict['attention_mask'].to(device)
        
    return input_dict



class D3MBERTMonitor(D3MBayesianMonitor):
    """Bayesian D3M Monitor for BERT tokenized features. Allows model class to take in both input_ids and attention_mask.
    Uses evaluation metrics provided by WILDS.

    Attributes:
        model (D3MAbstractModel): the model class, i.e. hypothesis class for the base classifier
        trainset (Dataset): torch training dataset
        valset (Dataset): torch validation dataset
        train_cfg (TrainConfig): TrainConfig object configuring all aspects of training
        device (torch.device): torch.device, cuda or cpu
    """
    
    def __init__(self, *args, **kwargs):
        super(D3MBERTMonitor, self).__init__(*args, **kwargs)
    
    def evaluate_model(self, loader, tqdm_enabled=False):
        running_loss = []
        running_acc = []
        with torch.no_grad():
            self.model.eval()
            for test_step, batch in enumerate(tqdm(loader, leave=False)):
                features, labels, *metadata = batch
                input_ids, attention_mask = features
                labels = labels.to(self.device)
                model_input = make_bert_input(input_ids, attention_mask, \
                    device=self.device, to_device=True)   
                out = self.model(model_input['input_ids'], model_input['attention_mask'])
                loss = out.val_loss_fn(labels)
                probs = out.predictive.probs
                acc = self.eval_acc(probs, labels).item()

                running_loss.append(loss.item())
                running_acc.append(acc)
                
        return running_loss, running_acc
    
    def get_f1_score(self, loader, tqdm_enabled=False):
        all_labels = []
        all_preds = []
        with torch.no_grad():
            self.model.eval()
            for test_step, batch in enumerate(tqdm(loader, leave=False)):
                features, labels, *metadata = batch
                input_ids, attention_mask = features
                labels = labels.to(self.device)
                model_input = make_bert_input(input_ids, attention_mask, \
                    device=self.device, to_device=True)   
                out = self.model(model_input['input_ids'], model_input['attention_mask'])
                probs = out.predictive.probs
                preds = torch.argmax(probs, dim=1).cpu()
                all_labels.append(labels.cpu())
                all_preds.append(preds)
        
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        metric = BinaryF1Score(threshold=0.5)
        metric.update(all_preds, all_labels)
        return metric.compute().item()
        
    
    
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
                input_ids, attention_mask = features
                self.optimizer.zero_grad()
                labels = labels.to(self.device)
                model_input = make_bert_input(input_ids, attention_mask,\
                    device=self.device, to_device=True)
                out = self.model(model_input['input_ids'], model_input['attention_mask'])
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
            
            # wandb loggingxwww
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
    
    
    def get_pseudolabels(self, dataset:Dataset):
        self.model.eval()
        dataset = dataset.to(self.device)
        with torch.no_grad():
            output = self.model.features(input_ids=dataset.input_ids, attention_mask=dataset.attention_mask)
            features = output.last_hidden_state[:,0,:]
            ll_dist = self.model.out_layer.logit_predictive(features)
            y_hat = torch.argmax(ll_dist.loc, 1)
        return y_hat
    
    def compute_max_dis_rate(self, dataset:Dataset, 
                             y_pseudo:torch.tensor, 
                             n_post_samples:int=5000, 
                             temperature:float=1.0):
        """Approximates the maximum disagreement rate among sampled weights.

        Args:
            dataset (Dataset): dataset object
            y_pseudo (torch.tensor): pseudolabels assigned by self.model
            n_post_samples (int, optional): number of posterior weights. Defaults to 5000.
            temperature (int, optional): softening of the logits. Defaults to 1.

        Returns:
            float: approximate maximum disagreement rate
        """
        self.model.eval()
        dataset = dataset.to(self.device)
        
        with torch.no_grad():
            output = self.model.get_features(input_ids=dataset.input_ids, \
                attention_mask=dataset.attention_mask)
            ll_dist = self.model.out_layer.logit_predictive(output)
            logits_samples = ll_dist.rsample(sample_shape=torch.Size([n_post_samples]))
            logits_samples = temperature_scaling(logits_samples, temperature)
            #y_hat = torch.argmax(logits_samples, -1)
            dist = torch.distributions.Categorical(logits=logits_samples)
            y_hat = dist.sample()
            y_tile = torch.tile(y_pseudo, (n_post_samples, 1)).to(self.device)
            dis_mat = (y_hat != y_tile)
            dis_rate = dis_mat.sum(dim=-1)/len(y_pseudo)
        return torch.max(dis_rate).item()
    
    def d3m_test(self, dataset:Dataset, data_sample_size:int=200, alpha=0.95, replace=True, *args, **kwargs) -> Tuple[float, bool]:
        """Given a dataset, computes the maximum disagreement rate as well as the OOD verdict.
        Used to both generate Phi and Deploy.

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
            sampled_dataset = dataset.sample(n=data_sample_size, replace=replace)
            y_pseudo = self.get_pseudolabels(sampled_dataset)
            max_dis_rate = self.compute_max_dis_rate(dataset=sampled_dataset, 
                                                     y_pseudo=y_pseudo, 
                                                     *args, **kwargs)


        return max_dis_rate, max_dis_rate >= np.quantile(self.Phi, alpha) if self.Phi != [] else 0 

    