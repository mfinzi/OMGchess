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
            conv2d(inter_channels,64,kernel_size=1,coords=coords))
    def forward(self,x):
        return self.net(x)

class ValueHead(nn.Module):
    def __init__(self,in_channels,fc_channels=1024,coords=True):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNrelu(in_channels,fc_channels//64,kernel_size=1,coords=coords),
            Expression(lambda u: u.view(-1,fc_channels)),
            FcBNrelu(fc_channels,fc_channels//2),
            nn.Linear(fc_channels//2,1),
            nn.Tanh())
    def forward(self,x):
        return self.net(x)

class simpleValueHead(nn.Module):
    def __init__(self,in_channels,coords=True):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNrelu(in_channels,in_channels,kernel_size=1,coords=coords),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(in_channels,1),
            nn.Tanh())

    def forward(self,x):
        return self.net(x)

class ChessResnet(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self,num_blocks=40,k=128,coords=True):
        super().__init__()
        self.net = nn.Sequential(
            conv2d(18,k,coords=coords),
            *[ResBlock(k,k,coords=coords) for _ in range(num_blocks)],
        )
        self.policy = ChessPolicyHead(k,64,coords=coords)
        self.value = simpleValueHead(k,coords=coords)

    def forward(self,x):
        common = self.net(x)
        return self.value(common),self.policy(common)