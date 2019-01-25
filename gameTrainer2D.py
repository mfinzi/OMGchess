import torch
import torch.nn as nn
from oil.utils.utils import Eval, cosLr, loader_to
from oil.trainer import Trainer


class GameTrainer2D(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def loss(self, minibatch):
        """ Standard cross-entropy loss """
        board,illegal_moves,target_value,target_action = minibatch
        value,logits_img = self.model(board)
        # N x 64 x (8 x 8)
        return nn.CrossEntropyLoss()(model(x),y)

    def batchAccuracy(self, minibatch, model = None):
        """ Evaluates the minibatch accuracy """
        if model is None: model = self.model
        with Eval(model), torch.no_grad():
            x, y = minibatch
            predictions = model(x).max(1)[1].type_as(y)
            accuracy = predictions.eq(y).cpu().data.numpy().mean()
        return accuracy
    
    def getAccuracy(self, loader, model = None):
        """ Gets the full dataset accuracy evaluated on the data in loader """
        num_correct, num_total = 0, 0
        for minibatch in loader:
            mb_size = minibatch[1].size(0)
            batch_acc = self.batchAccuracy(minibatch, model=model)
            num_correct += batch_acc*mb_size
            num_total += mb_size
        if not num_total: raise KeyError("dataloader is empty")
        return num_correct/num_total

    def logStuff(self, i, minibatch=None):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        step = i+1 + (self.epoch+1)*len(self.dataloaders['train'])

        metrics = {}
        #if minibatch: metrics['Train_Acc(Batch)'] = self.batchAccuracy(minibatch)
        try: metrics['Test_Acc'] = self.getAccuracy(self.dataloaders['test'])
        except KeyError: pass
        try: metrics['Dev_Acc'] = self.getAccuracy(self.dataloaders['dev'])
        except KeyError: pass
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(i,minibatch)