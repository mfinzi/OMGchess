import numpy as np
from numba import jit,njit,int32,float32,jitclass,void,boolean
import matplotlib.pyplot as plt
import time
import copy
import math
from math import sqrt
import sys
from mcts import MCTS

spec = [
    ('array', int32[:,:]),           
    ('col_length', int32[:]),
    ('num_moves_made', int32),
]
@jitclass(spec)
class Connect4Board(object):
    def __init__(self):
        self.array = np.zeros((6,7),dtype=int32)
        self.col_length = np.zeros(7,dtype=int32)
        self.num_moves_made = 0
    
    def copy(self, otherboard):
        self.array = np.copy(otherboard.array)
        self.col_length = np.copy(otherboard.col_length)
        self.num_moves_made = otherboard.num_moves_made
        
    def get_moves(self):
        moves = []
        for i in range(7):
            if self.array[-1,i]==0:
                moves.append(i)
        return moves#np.arange(7)[(self.board[-1]==0)]
        
    def color_to_move(self):
        return 2*(self.num_moves_made%2)-1
    
    def make_move(self,i):
        color = self.color_to_move()
        self.array[self.col_length[i],i] = color
        self.col_length[i] +=1
        self.num_moves_made +=1
        return self.move_won(i)*color
    
    def unmake_move(self,i):
        self.array[self.col_length[i]-1,i]=0
        self.col_length[i] -=1
        self.num_moves_made -=1
        
    #@staticmethod
    def inbounds(self,j,i):
        return (j<6) and (j>=0) and (i<7) and (i>=0)
    
    def move_won(self,i):
        j,i = self.col_length[i]-1,i
        color = self.array[j,i]
        if color==0:
            return False
        for (dj,di) in ((0,1),(1,0),(1,1),(1,-1)):
            connect_count = 1
            for k in range(1,4):
                nj,ni = j+k*dj,i+k*di
                if not self.inbounds(nj,ni) \
                    or self.array[nj,ni]!=color:break
                connect_count+=1
            for k in range(1,4):
                nj,ni = j-k*dj,i-k*di
                if not self.inbounds(nj,ni) \
                    or self.array[nj,ni]!=color:break
                connect_count+=1
            if connect_count >=4:
                return True
        return False
    
    def is_draw(self):
        return self.num_moves_made==42
    
    def reset(self):
        self.__init__()
        
    def data(self):
        return self.array[::-1]
    
    def show(self):
        plt.imshow(self.data())

def hashkey(board):
    return hash(board.array.tostring())

class Connect4Game(object):
    def __init__(self,move_first=True):
        self.engine = MCTS(Connect4Board)
        self.fig,self.ax = plt.subplots(1,1,figsize=(4,4))
        self.ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
        self.ax.set_xticks(np.arange(-.5, 7, 1), minor=True);
        self.ax.set_yticks(np.arange(-.5, 6, 1), minor=True);
        self.ppt = self.ax.imshow(self.engine.gameBoard.data(),vmin=-1,vmax=1)
        self.text_artist = self.ax.text(2,1,"",color='w')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()
        if not move_first: self.engine_move_update()
    
    def on_click(self,event):
        #plt.text(.5,.5,"arrg")
        if self.ax.in_axes(event):
            #self.engine.interrupt=True
            self.user_move_update(event)
            self.engine_move_update()
            #threading.thread(None,self.engine.ponder,args=(10,)).start()
            
    def user_move_update(self,event):
        user_move,j = self.get_click_coords(event)
        outcome = self.engine.make_move(user_move)
        if outcome: self.show_victory(outcome)
        #self.ax.plot(user_move,j,".r",markersize=4)
        self.ppt.set_data(self.engine.gameBoard.data())
        
    def engine_move_update(self):
        #self.text_artist.set_text("{}".format(self.engine.searchTree.num_visits))
        
        engine_move =self.engine.compute_move(1)
        #self.text_artist.set_text("{};{:1.2f}".format(Node.num_rollouts,self.engine.searchTree.win_ratio()))
        outcome = self.engine.make_move(engine_move)
        if outcome: self.show_victory(outcome)
        self.ppt.set_data(self.engine.gameBoard.data())
        #self.text_artist.set_text("{:1.2f}".format(self.engine.searchTree.win_ratio()))
        
            
    def show_victory(self,outcome):
        text = "WHITE WINS" if outcome==1 else "BLACK WINS"
        plt.text(5, 1.5, text, size=20,
             ha="right", va="top",
             bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
             )
    
    def get_click_coords(self,event):
        # Transform the event from display to axes coordinates
        imshape = self.ax.get_images()[0]._A.shape[:2]
        ax_pos = self.ax.transAxes.inverted().transform((event.x, event.y))
        rotate_left = np.array([[0,-1],[1,0]])
        i,j = (rotate_left@(ax_pos)*np.array(imshape)//1).astype(int)
        i,j = i%imshape[0],j%imshape[1]
        return j,i