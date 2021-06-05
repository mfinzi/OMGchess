
import os
import numpy as np
import torch
from oil.tuning.study import Study, train_trial
from oil.tuning.configGenerator import uniform,logUniform,sample_config
#import oil.augLayers as augLayers

from gameTrainer2D import GameTrainer2D, baseGameTrainTrial

logdir = os.path.expanduser('~/games/chess/runs/lr_study')
config_spec = {
    'num_epochs':8,
    'opt_config':{'lr':logUniform(.01,.3), 'momentum':.9, 'weight_decay':logUniform(1e-6,1e-4)},
    'trainer_config':{'log_dir':logdir,'value_weight':logUniform(.03,3)}
    }
cutout_study = Study(baseGameTrainTrial(strict=False),config_spec)
cutout_study.run(num_trials=15,max_workers=1)