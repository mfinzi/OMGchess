import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from oil.architectures.parts import conv2d,ConvBNrelu,FcBNrelu,ResBlock,DenseBlock
from chess_dataset import fen2tensor,legal_board_moves,move2class,class2move,legal_opponent_moves
import copy



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
            nn.Tanh())

    def forward(self,x):
        return self.net(x)[:,0]


class ChessNetwork(nn.Module,metaclass=Named):
    k = 4 # The number of boards in the history to be included
    @classmethod
    def encode(cls,board):
        """ Encodes a single chess board into a k move history
            and generates the legal moves and opponent moves.
            Method will destroy the original board."""
        board = copy.deepcopy(board)
        legal_moves = legal_board_moves(board).cuda()
        board.turn = not board.turn
        legal_opponent_moves = legal_board_moves(board).cuda()
        board.turn = not board.turn

        nn_boards = [board.start_tensor]*cls.k
        for i in range(min(len(board.move_stack),cls.k)):
            board.pop()
            nn_boards[cls.k-i-1] = fen2tensor(board.fen())
        nn_boards = torch.cat(nn_boards,dim=0).cuda() # no batch dim yet
        # For now the legal moves and opponent moves are left separate
        return nn_boards,legal_moves,legal_opponent_moves

    def predict(self,x):
        values,logits = self(x)
        return values, F.softmax(logits,dim=1)
        
    def forward(self,boards,legal_moves): #switch from individual tensors here to one tuple=> encoded input
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
            conv2d(18*self.k+64*4,k,coords=coords),
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



class ChessDRN(ChessNetworkWopp):
    def __init__(self,k=128,num_blocks=40,drop_rate=.3,coords=True):
        m = num_blocks//2
        p = drop_rate
        super().__init__()
        self.net = nn.Sequential( #Idea, remove intermediate grid artifacts via intermediate conv2d
            conv2d(18*self.k+64*4,k,coords=coords),
            *[ResBlock(  k,  k,dilation=1,coords=coords,drop_rate=p) for _ in range(m)],
            ConvBNrelu(  k,  k,dilation=1,coords=coords), # Problem because of duplicated conv? harder to train?
            *[ResBlock(  k,  k,dilation=2,coords=coords,drop_rate=p) for _ in range(m//2)],
            ConvBNrelu(  k,  k,dilation=1,coords=coords),
            *[ResBlock(  k,  k,dilation=4,coords=coords,drop_rate=p) for _ in range(m//4)],
            ConvBNrelu(  k,  k,dilation=2,coords=coords),
            ConvBNrelu(  k,  k,dilation=1,coords=coords),
        )
        self.policy = ChessPolicyHead(k,64,coords=coords)
        self.value = SimpleValueHead(k,coords=coords)#ValueHead(k,1024,coords=coords)#
        print("{}M Parameters".format(sum(p.numel() for p in self.net.parameters() if p.requires_grad)/10**6))