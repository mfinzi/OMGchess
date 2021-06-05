import numpy as np
from numba import jit,njit,int32,float32,jitclass,void,boolean,int64
import matplotlib.pyplot as plt
import time
import copy
import math
from math import sqrt
import sys
from mcts import MCTS

coords = np.mgrid[:9,:9].T
spec = [
    ('array', int32[:,:]),     
    ('num_moves_made', int32),
]
@jitclass(spec)
class GoBoard(object):
    
    def __init__(self):
        self.array = np.zeros((9,9),dtype=int32)
        self._color_to_move = 1

    def copy(self, otherboard):
        self.array = np.copy(otherboard.array)
        self.color_to_move = otherboard.color_to_move

    def get_moves(self):
        # TODO: deal with recapture
        return coords[self.array==0]
        
    def color_to_move(self):
        return self._color_to_move
    
    def make_move(self,coords):
        color = self.color_to_move()
        i,j = coords
        self.array[i,j] = color
        self._color_to_move *=-1
        return self.move_won(i)*color
        
    #@staticmethod
    def inbounds(self,j,i):
        return (j<6) and (j>=0) and (i<7) and (i>=0)
    
    def amove_won(self):
        for i in range(7):
            if self.move_won(i):
                return True
        return False

    def move_won(self,i):
        return False
    
    def is_draw(self):
        return self.num_moves_made==42
    
    def reset(self):
        self.__init__()
        
    def data(self):
        return self.array[::-1]
    
    def show(self):
        plt.imshow(self.data())



class GoGame(Connect4Game):
    Board = GoBoard