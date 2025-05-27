import os
os.environ['HYDRA_FULL_ERROR'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import wandb
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)
import fcntl

from d3m.monitors import D3MBayesianMonitor, D3MFullInformationMonitor, D3MBERTMonitor
from d3m.models import ConvModel, MLPModel, ResNetModel, BERTModel

import torch
import torch.nn as nn
#import torch.multiprocessing as mp
import numpy as np
from experiments.utils import get_datasets, get_configs


torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# base models
base_models = {
    'cifar10': ConvModel,
    'uci': MLPModel,
    'synthetic': MLPModel,
    'camelyon17': ResNetModel,
    'civilcomments': BERTModel,
}


monitors = {
    'bayesian': D3MBayesianMonitor,
    'fi': D3MFullInformationMonitor,
    'bert': D3MBERTMonitor,
}


@hydra.main(config_path='configs/', config_name='civilcomments', version_base='1.2')
def main(args:DictConfig):
    # =========================================================
    # ========================Seeding==========================
    # =========================================================
    
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)
    torch.random.manual_seed(RANDOM_SEED)
    
    # =========================================================
    # =============Print All Configurations====================
    # =========================================================
    
    print("Hydra Configuration:")
    print(OmegaConf.to_yaml(args))
    
    # ==================Initialization=========================
    # =========================================================
    
    ''' wandb config initialization '''
    if args.wandb_enabled:
        wandb.config = OmegaConf.to_container(
            args, resolve=True, throw_on_missing=True
        )
        
        run = wandb.init(project=args.wandb_cfg.project,
                        settings=wandb.Settings(start_method='thread') # for hydra
                        )
        
        run.config.update({'GPU': torch.cuda.get_device_name()})
    
    print('GPU: ', torch.cuda.get_device_name())
    
    ''' Get datasets '''
    dataset = get_datasets(args) 
    
    ''' Get config objects '''
    model_config, train_config = get_configs(args)
    
    ''' Build model and monitor '''
    base_model = base_models[args.dataset.name](model_config, train_size=len(dataset['train']))
    monitor = monitors[args.monitor_type](
        model=base_model,
        trainset=dataset['train'],
        valset=dataset['valid'],
        train_cfg=train_config,
        device=device,
    )
    
    ''' Log random seed '''
    if args.wandb_enabled:
        wandb.log({'seed': args.seed})
    
    # =========================================================
    # ==============Base Classifier Training===================
    # =========================================================
    
    ''' Load base model if pretrained, else train '''
    if args.from_pretrained:
        base_model.load_state_dict(torch.load(os.path.join('saved_weights', f'{args.dataset.name}.pth')))
    else:
        # make ood testloader
        ood_testloader = torch.utils.data.DataLoader(dataset['d3m_ood'],
                                                     batch_size=train_config.batch_size,
                                                     shuffle=False,
                                                     num_workers=train_config.num_workers,
                                                     pin_memory=train_config.pin_memory)
        monitor.train_model(tqdm_enabled=True, testloader=ood_testloader)
        os.makedirs('saved_weights', exist_ok=True)
        torch.save(monitor.model.state_dict(), os.path.join('saved_weights', f'{args.dataset.name}.pth'))
    
    # =========================================================
    # ===================D3M Training=======================
    # =========================================================
    
    ''' Pretrain the disagreement distribution Phi '''
    monitor.pretrain_disagreement_distribution(dataset=dataset['d3m_train'],
                                               n_post_samples=args.d3m.n_post_samples,
                                               data_sample_size=args.d3m.data_sample_size,
                                               Phi_size=args.d3m.Phi_size, 
                                               temperature=args.d3m.temp,
                                               )
    
    ''' wandb log Phi statistics '''
    if args.wandb_enabled:
        wandb.log({
            'Phi-mean': np.mean(monitor.Phi),
            'Phi-std': np.std(monitor.Phi),
            'Phi-med': np.median(monitor.Phi)
        })
        
    # =========================================================
    # ===================D3M Testing========================
    # =========================================================
    
    ''' Test TPR/FPR on all datasets '''
    stats = {}
    dis_rates = {}
    for k,dataset in {
        'd3m_train': dataset['d3m_train'],
        'd3m_id': dataset['d3m_id'],
        'd3m_ood': dataset['d3m_ood']
    }.items():
        rate, max_dis_rates = monitor.repeat_tests(n_repeats=args.d3m.n_repeats,
                                      dataset=dataset, 
                                      n_post_samples=args.d3m.n_post_samples,
                                      data_sample_size=args.d3m.data_sample_size,
                                      temperature=args.d3m.temp
                                      )
        print(f"{k}: {rate}")
        stats[k] = rate
        dis_rates[k] = (np.mean(max_dis_rates), np.std(max_dis_rates))

    # =========================================================
    # ===================Logging Results=======================
    # =========================================================
    
    if args.wandb_enabled:
        ''' wandb log statistics '''
        wandb.log({
            'fpr_train': stats['d3m_train'],
            'fpr_id': stats['d3m_id'],
            'tpr': stats ['d3m_ood']
        })
        wandb.log({
            'dr_train': dis_rates['d3m_train'],
            'dr_id': dis_rates['d3m_id'],
            'dr_ood': dis_rates['d3m_ood']
        })
    
    if args.self_log:
        ''' Self-logging initialization '''
        logger = {}
        log_dir = 'results/'
        os.makedirs(log_dir, exist_ok=True)
        
        logger['seed'] = args.seed
        logger['data_sample_size'] = args.d3m.data_sample_size
        
        ''' self-log statistics '''
        logger['fpr_train'] = stats['d3m_train']
        logger['fpr_id'] = stats['d3m_id']
        logger['tpr'] = stats['d3m_ood']
        
        ''' write self-log to file '''
        csv_path = os.path.join(log_dir, f'results_{args.dataset.name}_{args.d3m.data_sample_size}_final.csv')
        csv_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            df = pd.DataFrame([logger])
            if not csv_exists:
                df.to_csv(f, index=False)
            else:
                df.to_csv(f, mode='a', header=False, index=False)
            fcntl.flock(f, fcntl.LOCK_UN)
            
    return 0


if __name__ == '__main__':
    #mp.set_start_method('spawn', force=True)
    main()