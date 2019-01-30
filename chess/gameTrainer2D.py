import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from oil.utils.utils import Eval, cosLr
from oil.model_trainers import Trainer,Classifier,Regressor
from oil.utils.mytqdm import tqdm

class GameTrainer2D(Trainer):
    """ """
    def __init__(self,*args,value_weight=.3,**kwargs):
        super().__init__(*args,**kwargs)
        self.hypers['value_weight'] = value_weight

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        boards,legal_moves,target_values,target_actions = minibatch
        values,logits = self.model(boards,legal_moves) # N x 4096
        CE = F.cross_entropy(logits,target_actions)
        MSE = ((values-target_values)**2).mean()#F.mse_loss(values,target_values)
        return (CE + self.hypers['value_weight']*MSE)/(1+self.hypers['value_weight'])

    def _metrics(self,minibatch):
        boards,legal_moves,target_values,target_actions = minibatch
        values,logits = self.model(boards,legal_moves)
        preds = logits.max(1)[1].type_as(target_actions)
        acc = preds.eq(target_actions).cpu().data.numpy().mean()
        mse = ((values-target_values)**2).mean().cpu().data.numpy()
        value2cp = self.dataloaders['train'].dataset.value2cp
        cp_rms = np.sqrt((value2cp(values.cpu().data.numpy())-value2cp(target_values.cpu().data.numpy())).mean())
        return np.array([acc,mse,cp_rms])

    def metrics(self,loader):
        acc,mse,cp_rms = self.evalAverageMetrics(loader,self._metrics)
        return {'Acc':acc,'MSE':mse,'CentipawnRMS':cp_rms}
    #todo: add rms centipawn score, unify metrics
    def evalELO(self,loader):
        pass


    

# Convenience function for that covers a common use case of training the model using
#   the cosLr schedule, and logging the outcome and returning the results
import os
from oil.utils.utils import LoaderTo, cosLr, recursively_update#loader_to,to_device_layer
from oil.tuning.study import train_trial
from chess_dataset import ChessDataset
from chess_network import ChessResnet
from torch.utils.data import DataLoader
# from oil.datasetup.dataloaders import getLabLoader
# from oil.datasetup.datasets import CIFAR10
# from oilarchitectures.img_classifiers import layer13s
import collections


def makeSimpleTrainer(config):
    cfg = {
        'dataset': 'chess_3000k_0.2s',
        'datadir': os.path.expanduser('~/games/chess/data/'),
        'network':ChessResnet,'net_config': {'coords':True,'num_blocks':20,'k':128},
        'bs': 128, 
        'opt_config':{'lr':.1, 'momentum':.9, 'weight_decay':1e-5,'nesterov':True},#'opt_constr':lambda params: torch.optim.Adam(params,lr=1e-3),#lambda params: torch.optim.SGD(params,**{'lr':.03, 'momentum':.9, 'weight_decay':1e-4}),
        'num_epochs':30,'trainer_config':{'value_weight':.01},
        }
    
    recursively_update(cfg,config)
    lr_sched = cosLr(cfg['num_epochs'])
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
    opt_constr = lambda params: torch.optim.SGD(params, **cfg['opt_config'])
    return GameTrainer2D(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])

def baseGameTrainTrial(strict=False):
    return train_trial(makeSimpleTrainer,strict)

if __name__=='__main__':
    num_epochs = 30
    logdir = os.path.expanduser('~/games/chess/runs/end_encoding_fullValueHead/')
    trainer = makeSimpleTrainer({'num_epochs':num_epochs,'trainer_config':{'log_dir':logdir}})
    for i in tqdm(range(num_epochs),desc='epochs'):
        trainer.train(1)
        trainer.logger.save_object(trainer,suffix='checkpoints/c{}.trainer'.format(trainer.epoch))
    