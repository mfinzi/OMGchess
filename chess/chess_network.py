import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from oil.architectures.parts import conv2d,ConvBNrelu,FcBNrelu,ResBlock





class ChessPolicyHead(nn.Module):
    """ A simplified version of the alphazero paper
        only 64 move outputs, one for each square,
        underpromotions are not considered,
        encoding is: TODO """
    def __init__(self,in_channels,inter_channels,coords):
        super().__init__()
        self.net = nn.Sequential(
            conv2d(in_channels,inter_channels,kernel_size=1,coords=coords),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            conv2d(inter_channels,64,kernel_size=1,coords=coords),
            Expression(lambda u: u.view(u.shape[0],-1)))
    def forward(self,x):
        return self.net(x)

class ValueHead(nn.Module):
    def __init__(self,in_channels,fc_channels=1024,coords=True):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNrelu(in_channels,fc_channels//64,kernel_size=1,coords=coords),
            Expression(lambda u: u.view(u.size(0),fc_channels)),
            FcBNrelu(fc_channels,fc_channels//2),
            nn.Linear(fc_channels//2,1),
            nn.Tanh())
    def forward(self,x):
        return self.net(x)[:,0]

class SimpleValueHead(nn.Module):
    def __init__(self,in_channels,coords=True):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNrelu(in_channels,in_channels,kernel_size=1,coords=coords),
            ConvBNrelu(in_channels,in_channels,kernel_size=1,coords=coords),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(in_channels,1),
            nn.Tanh())

    def forward(self,x):
        return self.net(x)[:,0]

class ChessResnet(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self,num_blocks=40,k=128,coords=True):
        super().__init__()
        self.net = nn.Sequential(
            conv2d(18+64+64,k,coords=coords),
            *[ResBlock(k,k,coords=coords) for _ in range(num_blocks)],
        )
        self.policy = ChessPolicyHead(k,64,coords=coords)
        self.value = SimpleValueHead(k,coords=coords)#ValueHead(k,1024,coords=coords)#

    def forward(self,boards,legal_moves):
        move_end_encoding = legal_moves.view(-1,64,8,8).float()
        move_start_encoding = legal_moves.view(-1,64,64).permute(0,2,1).view(-1,64,8,8).float()
        input_features = torch.cat([boards,move_end_encoding,move_start_encoding],dim=1)
        common = self.net(input_features)
        value = self.value(common)
        logits = self.policy(common)
        logits[~legal_moves] = -1e10
        return value,logits