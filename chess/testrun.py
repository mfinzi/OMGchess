
import os
import numpy as np
import torch
from oil.tuning.study import Study, train_trial
from oil.tuning.configGenerator import uniform,logUniform,sample_config
from oil.utils.utils import LoaderTo, cosLr, recursively_update
from oil.tuning.study import train_trial
from chess_dataset import ChessDataset
from chess_network import ChessResnet
from torch.utils.data import DataLoader
#import oil.augLayers as augLayers

from gameTrainer2D import GameTrainer2D, baseGameTrainTrial

logdir = os.path.expanduser('~/games/chess/runs/adamtest')
adam_config = {
    'trainer_config':{'log_suffix':'adam/'},
    'optimizer':torch.optim.Adam,
    'opt_config':{'lr':2e-3},
    'num_epochs':8,
}
sgd_config = {
    'trainer_config':{'log_suffix':'sgd/'},
    'optimizer':torch.optim.SGD,
    'opt_config':{'lr':.1,'momentum':.9,'weight_decay':2e-6,'nesterov':True},
    'num_epochs':8,
}

def makeTrainer(config):
    cfg = {
        'dataset': 'chess_3000k_0.2s',
        'datadir': os.path.expanduser('~/games/chess/data/'),
        'bs': 128,
        'network':ChessResnet,'net_config': {'coords':True,'num_blocks':20,'k':128},
        'trainer_config':{'log_dir':logdir,'value_weight':2.5}
        }
    cfg = recursively_update(cfg,config)
    lr_sched = cosLr()
    trainset = ChessDataset(cfg['datadir']+cfg['dataset']+'_train_0.pkl')
    train_small = ChessDataset(cfg['datadir']+cfg['dataset']+'_trainsmall.pkl')
    val = ChessDataset(cfg['datadir']+cfg['dataset']+'_val.pkl')
    device = torch.device('cuda')
    fullCNN = cfg['network'](**cfg['net_config']).to(device)
    dataloaders = {}
    dataloaders['train'] = DataLoader(trainset,batch_size=cfg['bs'],
                            shuffle=True,drop_last=True,pin_memory=True,num_workers=4)
    dataloaders['train_'] = DataLoader(train_small,batch_size=cfg['bs'],shuffle=False)
    dataloaders['val'] = DataLoader(val,batch_size=cfg['bs'],shuffle=False)
    dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
    opt_constr = lambda params: cfg['optimizer'](params, **cfg['opt_config'])
    return GameTrainer2D(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])

Trial = train_trial(makeTrainer,strict=True)
Trial(adam_config)
Trial(sgd_config)