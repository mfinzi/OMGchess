import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from oil.architectures.parts import conv2d,ConvBNrelu,FcBNrelu,ResBlock,DenseBlock





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

class DualPolicyHead(nn.Module):
    def __init__(self,in_channels,inter_channels,coords):
        super().__init__()
        self.net1 = nn.Sequential(
            conv2d(in_channels,inter_channels,kernel_size=1,coords=coords),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            conv2d(inter_channels,64,kernel_size=1,coords=coords),
            Expression(lambda u: u.view(u.shape[0],-1)))
        self.net2 = nn.Sequential(
            conv2d(in_channels,inter_channels,kernel_size=1,coords=coords),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            conv2d(inter_channels,64,kernel_size=1,coords=coords),
            Expression(lambda u: u.view(-1,64,64).permute(0,2,1).contiguous().view(-1,64*64)))
    def forward(self,x):
        return self.net1(x) + self.net2(x)

class ValueHead(nn.Module):
    def __init__(self,in_channels,fc_channels=1024,coords=True):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNrelu(in_channels,fc_channels//64,kernel_size=1,coords=coords),
            Expression(lambda u: u.view(u.size(0),fc_channels)),
            FcBNrelu(fc_channels,fc_channels//2),
            nn.Linear(fc_channels//2,1),
            )#nn.Tanh())
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
            )#nn.Tanh())

    def forward(self,x):
        return self.net(x)[:,0]


class ChessNetwork(nn.Module,metaclass=Named):
    def forward(self,boards,legal_moves):
        move_end_encoding = legal_moves.view(-1,64,8,8).float()
        move_start_encoding = legal_moves.view(-1,64,64).permute(0,2,1).view(-1,64,8,8).float()
        input_features = torch.cat([boards,move_end_encoding,move_start_encoding],dim=1)
        common = self.net(input_features)
        value = self.value(common)
        logits = self.policy(common)
        logits[~legal_moves] = -1e10
        return value,logits

class ChessNetworkWopp(ChessNetwork):
    def forward(self,boards,legal_moves,opponent_legal_moves):
        opp_end_encoding = opponent_legal_moves.view(-1,64,8,8).float()
        opp_start_encoding = opponent_legal_moves.view(-1,64,64).permute(0,2,1).view(-1,64,8,8).float()
        boards_and_opp_moves = torch.cat([boards,opp_end_encoding,opp_start_encoding],dim=1)
        return super().forward(boards_and_opp_moves,legal_moves)

class ChessResnet(ChessNetworkWopp):
    def __init__(self,num_blocks=40,k=128,drop_rate=0,coords=True):
        super().__init__()
        self.net = nn.Sequential(
            conv2d(18*4+64*4,k,coords=coords),
            *[ResBlock(k,k,drop_rate=drop_rate,coords=coords) for _ in range(num_blocks)],
        )
        self.policy = DualPolicyHead(k,64,coords=coords)
        self.value = SimpleValueHead(k,coords=coords)#ValueHead(k,1024,coords=coords)#
        print("{}M Parameters".format(sum(p.numel() for p in self.net.parameters() if p.requires_grad)/10**6))


class ChessDensenet(ChessNetwork):
    def __init__(self,M=5,N=15,k=24,drop_rate=0,coords=True):
        super().__init__()
        dense_layers = []
        inplanes = 192
        for i in range(M):
            dense_layers.append(DenseBlock(inplanes,k,N,drop_rate,coords))
            inplanes = (inplanes + N*k)//2

        self.net = nn.Sequential(
            conv2d(18+64*4,192,coords=coords),
            *dense_layers,
        )
        self.policy = ChessPolicyHead(inplanes,64,coords=coords)
        self.value = SimpleValueHead(inplanes,coords=coords)#ValueHead(k,1024,coords=coords)#
        print("{}M Parameters".format(sum(p.numel() for p in self.net.parameters() if p.requires_grad)/10**6))

