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
                board_array[piece2plane[char],7-i,j]=1
                j+=1
    # encode castling rights
    for i in range(4): 
        board_array[i+12] = ('KQkq'[i] in castling)
    # encode color and move count
    board_array[16] = (color == 'w')
    board_array[17] = mvnum
    return torch.from_numpy(board_array).float()

def legal_moves(fen_string):
    board = chess.Board(fen_string)
    return legal_board_moves(board)

def legal_board_moves(board):
    # Currently encoded by end location as 8x8
    legal_ids = [move2class(move.uci()) for move in board.legal_moves]
    illegal_mask = np.ones(64*64)
    illegal_mask[legal_ids] = 0
    return ~torch.from_numpy(illegal_mask).byte()

def move2class(move_string):
    c1,r1,c2,r2 = move_string[:4] # underpromotions are ignored
    i,j,k,l = ord(c1)-ord('a'),int(r1)-1,ord(c2)-ord('a'),int(r2)-1
    n,m = 8*j+i, 8*l+k #Fix misordering here
    class_index = 64*n + m
    return class_index

def class2move(class_id):
    start, end = class_id//64,class_id%64
    row1, col1 = start//8, start%8 #Fix misordering here
    row2, col2 = end//8, end%8
    uci_string = chr(col1+ord('a'))+str(row1+1)+chr(col2+ord('a')) + str(row2+1)
    return uci_string # what about promotions?

def cp2value(centipawn_score):
    # Who doesnt love magic numbers?
    return (np.arctan(centipawn_score/290.68)/1.56).astype(np.float32)

def value2cp(value):
    return 290.68*np.tan(value*1.56)

class ChessDataset(Dataset,metaclass=Named):
    class_weights=None
    def __init__(self,filepath):
        with open(filepath, 'rb') as file:
            self.pgns = dill.load(file)

    def __getitem__(self,index):
        fen, score, move = self.pgns[index]
        # board_tensor, illegal_move_list, value, class_index
        return fen2tensor(fen), legal_moves(fen),\
               cp2value(score), move2class(move)

    def __len__(self):
        return len(self.pgns)

    @staticmethod
    def cp2value(centipawn_score):
        # Who doesnt love magic numbers?
        return (np.arctan(centipawn_score/290.68)/1.56).astype(np.float32)
    @staticmethod
    def value2cp(value):
        return 290.68*np.tan(value*1.56)