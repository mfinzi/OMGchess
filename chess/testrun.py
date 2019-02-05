
import os
import numpy as np
import torch
from oil.tuning.study import Study, train_trial
from oil.tuning.configGenerator import uniform,logUniform,sample_config
from oil.utils.utils import LoaderTo, cosLr, recursively_update
from oil.tuning.study import train_trial
from chess_dataset import ChessDataset,ChessDatasetWOpp
from chess_network import ChessResnet,ChessDensenet
from torch.utils.data import DataLoader
#import oil.augLayers as augLayers

from gameTrainer2D import GameTrainer2D, baseGameTrainTrial

logdir = os.path.expanduser('~/games/chess/runs/drop2')
adam_config = {
    #'trainer_config':{'log_suffix':'adam/'},
    'optimizer':torch.optim.Adam,
    'opt_config':{'lr':2e-3},
    'num_epochs':8,
    'network':ChessResnet,'net_config': {'coords':True,'num_blocks':20,'k':128,'drop_rate':.3},
    #'network':ChessDensenet,'net_config': {'M':5,'N':20,'k':20,'drop_rate':0,'coords':True},
}
sgd_config = {
    'trainer_config':{'log_suffix':'sgd/'},
    'optimizer':torch.optim.SGD,
    'opt_config':{'lr':.2,'momentum':.9,'weight_decay':2e-6,'nesterov':True},
    'num_epochs':8,
    'trainer_config':{'value_weight':1}
}

def makeTrainer(config):
    cfg = {
        'dataset': 'chess_3001k_0.2s',
        'datadir': os.path.expanduser('~/games/chess/data/'),
        'bs': 128,
        'trainer_config':{'log_dir':logdir,'value_weight':2.5}
        }#'network':ChessResnet,'net_config': {'coords':True,'num_blocks':20,'k':128},
    cfg = recursively_update(cfg,config)
    lr_sched = cosLr()
    trainset = ChessDatasetWOpp(cfg['datadir']+cfg['dataset']+'_train_0.pkl')
    train_small = ChessDatasetWOpp(cfg['datadir']+cfg['dataset']+'_trainsmall.pkl')
    val = ChessDatasetWOpp(cfg['datadir']+cfg['dataset']+'_val.pkl')
    device = torch.device('cuda')
    fullCNN = cfg['network'](**cfg['net_config']).to(device)
    dataloaders = {}
    dataloaders['train'] = DataLoader(trainset,batch_size=cfg['bs'],
                            shuffle=True,drop_last=True,pin_memory=True,num_workers=2)
    dataloaders['train_'] = DataLoader(train_small,batch_size=cfg['bs'],shuffle=False)
    dataloaders['val'] = DataLoader(val,batch_size=cfg['bs'],shuffle=False)
    dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
    opt_constr = lambda params: cfg['optimizer'](params, **cfg['opt_config'])
    return GameTrainer2D(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])

Trial = train_trial(makeTrainer,strict=True)
Trial(adam_config)
#Trial(sgd_config)

# Completed Improvements:
# Coordinate convolutions in all layers (unknown)
# Feed legal moves as input to the network (major boost)
# Use both start and end legal move encodings for input (small improvement?)
# No Improvement: remove tanh on value network, train on cp value directly (weights too much on extreme states?)
# Minor to no Improvement: Add opponent move encoding to input features (yields worse or similar accs?)
# Dual encoding policy network (+.7% acc)
# Encode partial move history into the input tensors (+x% acc)
# Dropout p=0.3: +.5% acc
# Why does Adam work better?

# TODO: Replace resnet backbone with a densenet  (in progress, helps but more so with value function)
# TODO: Get SWA setup and working
# TODO: Yarin Gal's multitask uncertainty loss for balancing policy & value
# TODO: Add in a FiLM layer using (to_move,num_moves,castling rights features)

# TODO: Weight sharing with repeating layers (aka RNN) for planning (investigate CTC)
# TODO: Add (flip board, swap white for black pieces and tomove, negate cp) data aug (only 1.6x data though?)

# TODO: Primitive elo evaluation
# TODO: Measure speed of inference as function of batch size
# TODO: Move to lower precision inference (16 bits)
# TODO: Move to TensorRT with onnx

# Asynchronous MCTS
# Base cython implementation
# + Multithreading (GPU queue)
# + Transposition table (get zobrist keys)
# + Opening Book
# + Endgame database

# Visualizations
# Graph of acc,mse value pairs vs elo
# Table of improvements, e.g. + dual-head: + 1.5 acc, -.5 mse, +200 elo
# Detailed plot of elo vs temperature
# Visualization of the search tree: histogram for branch depth
# Graph of performence vs size of dataset 10^5 -> 10^8

# Extensions
# Static elo evaluation task:
# Use trained network & train from scratch
# Try to predict elos of black and white players
# Alternative approach with interactive agent
# CPNS feature visualization