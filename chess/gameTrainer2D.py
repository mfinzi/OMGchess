import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from oil.utils.utils import Eval, cosLr
from oil.model_trainers import Trainer,Classifier,Regressor


class GameTrainer2D(Trainer):
    """ """
    #metrics = {**Classifier.metrics,**Regressor.metrics}
    def __init__(self,*args,value_weight=.01,**kwargs):
        super().__init__(*args,**kwargs)
        self.hypers['value_weight'] = value_weight

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        boards,illegal_moves,target_values,target_actions = minibatch
        values,logits = self.model(boards)
        # convert N x 64 x 8 x 8 to N x 4096
        #logits = logits_img.view(logits_img.shape[0],-1)
        # Mask out illegal move options
        logits[illegal_moves] = -1e10
        # print(logits.type())
        # print(target_actions.type())
        # print([a.type() for a in minibatch])
        # Check that size average and reduce match a0 paper
        CE = F.cross_entropy(logits,target_actions)
        MSE = ((values-target_values)**2).mean()#F.mse_loss(values,target_values)
        return (CE + self.hypers['value_weight']*MSE)/(1+self.hypers['value_weight'])
    
    def logStuff(self,i,minibatch=None):
        step = i+1 + (self.epoch+1)*len(self.dataloaders['train'])
        metrics = {}
        metrics['Train_Acc'],metrics['Train_MSE'] =\
            self.evalAverageMetrics(self.dataloaders['train'],self._metrics)
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(i,minibatch)

    def _metrics(self,minibatch):
        boards,illegal_moves,target_values,target_actions = minibatch
        values,logits = self.model(boards)
        logits[illegal_moves] = -1e10
        preds = logits.max(1)[1].type_as(target_actions)
        acc = preds.eq(target_actions).cpu().data.numpy().mean()
        mse = ((values-target_values)**2).mean().cpu().data.numpy()
        return np.array([acc,mse])

    def evalELO(self,loader):
        pass


    

# Convenience function for that covers a common use case of training the model using
#   the cosLr schedule, and logging the outcome and returning the results
import os
from oil.utils.utils import LoaderTo#loader_to,to_device_layer
from oil.tuning.study import train_trial
from chess_dataset import ChessDataset
from chess_network import ChessResnet
from torch.utils.data import DataLoader
# from oil.datasetup.dataloaders import getLabLoader
# from oil.datasetup.datasets import CIFAR10
# from oilarchitectures.img_classifiers import layer13s
import collections

def recursively_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursively_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def baseGameTrainTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': ChessDataset,'network':ChessResnet,'net_config': {'coords':False},
            'bs': 128, 
            'opt_constr':lambda params: torch.optim.SGD(params,**{'lr':.4, 'momentum':.9, 'weight_decay':1e-4}),
            'num_epochs':100,'trainer_config':{},
            }
        cfg['lr_sched'] = cosLr(cfg['num_epochs'])
        recursively_update(cfg,config)
        trainset = cfg['dataset'](os.path.expanduser('~/games/chess/chess_train.pkl'))
        device = torch.device('cuda')
        fullCNN = torch.nn.Sequential(
            cfg['network'](**cfg['net_config']).to(device)
        )
        dataloaders = {}
        dataloaders['train'] = DataLoader(trainset,batch_size=cfg['bs'],shuffle=True)
        dataloaders['dev'] = DataLoader(trainset,batch_size=cfg['bs'],shuffle=False)
        dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
        #opt_constr = , **cfg['opt_config'])
        return GameTrainer2D(fullCNN,dataloaders,cfg['opt_constr'],cfg['lr_sched'],**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)