import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as ds
import torch.nn as nn
import numpy as np
from oil.utils.utils import Named
import dill
import chess


def fen2tensor(fen_string):
    board_array = np.zeros((18,8,8))
    brd,color,castling,enps,hc,mvnum = fen_string.split(' ')
    # Encode p1 and p2 pieces,
    piece2plane = {'PRNBQKprnbqk'[i]:i for i in range(12)}
    for i,row in enumerate(brd.split('/')):
        j=0
        for char in row:
            if char.isdigit():
                j+=int(char)
            else:
                board_array[piece2plane(char),i,j]=1
                j+=1
    # encode castling rights
    for i in range(4): 
        board_array[i+12] = ('KQkq'[i] in castling)
    # encode color and move count
    board_array[16] = (color == 'w')
    board_array[17] = mvnum
    return torch.from_numpy(board_array)

def illegal_moves(fen_string):
    board = chess.board(fen_string)
    legal_ids = [move2class(move.uci()) for move in board.legal_moves]
    illegal_moves = np.delete(np.arange(64*64),legal_ids)
    return illegal_moves

def move2class(move_string):
    # Check that img reshape orders correctly
    c1,r1,c2,r2 = move_string
    i,j,k,l = ord(c1)-ord('a'),int(r1)-1,ord(c2)-ord('a'),int(r2)-1
    n,m = 8*i+j, 8*l+k
    class_index = 64*n + m 
    return class_index

def cp2value(centipawn_score):
    # Who doesnt love magic numbers?
    return np.arctan(centipawn_score/290.68)/1.55

class ChessDataset(Dataset):
    def __init__(self,filepath):
        with open(filepath, 'rb') as file:
            self.pgns = dill.load(file)

    def __getitem__(self,index):
        fen, (score, move) = self.pgns[index]
        return fen2tensor(fen), illegal_moves(fen),\
               cp2value(score), move2class(move)

    def __len__(self):
        return len(self.pgns)