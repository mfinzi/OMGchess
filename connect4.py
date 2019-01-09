import numpy as np
from numba import jit,njit,int32,float32,jitclass,void,boolean,int64
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
    
    def amove_won(self):
        for i in range(7):
            if self.move_won(i):
                return True
        return False

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

spec = [
    ('p1', int64),
    ('p2', int64),           
    ('col_lengths', int32),
    ('num_moves_made', int32),
]
@jitclass(spec)
class Connect4BitBoard(object):
    def __init__(self):
        self.p1 = 0 # encoded 0b[col7]00[col2]00..00[col1]
        self.p2 = 0
        self.col_lengths = 0 # encoded 3 bits each 0b[010][111][000]...[]
        self.num_moves_made = 0
    
    def copy(self, otherboard):
        self.p1 = otherboard.p1
        self.p2 = otherboard.p2
        self.col_lengths = otherboard.col_lengths
        self.num_moves_made = otherboard.num_moves_made
        
    def get_moves(self):
        filled = (self.p1|self.p2)>>5
        moves = []
        for i in range(7):
            if not filled&0x1:
                moves.append(i)
            filled = filled>>8
        return moves
        
    def color_to_move(self):
        return 2*(self.num_moves_made%2)-1
    
    def make_move(self,i):
        color = self.color_to_move()
        bitboard = self.p1 if color==-1 else self.p2
        col_length = (self.col_lengths>>(3*i))&7
        bitboard |= ((1<<col_length)<<(8*i))
        if color==-1: self.p1=bitboard
        else: self.p2=bitboard
        self.col_lengths += 1<<(3*i)
        self.num_moves_made += 1
        return self.move_won(i)*color
    
    def amove_won(self):
        color = self.color_to_move()
        bitboard = self.p2 if color==-1 else self.p1
        # Check \
        temp_bboard = bitboard & (bitboard >> 7)
        if(temp_bboard & (temp_bboard >> 2 * 7)):
            return True
        # Check -
        temp_bboard = bitboard & (bitboard >> 8)
        if(temp_bboard & (temp_bboard >> 2 * 8)):
            return True
        # Check /
        temp_bboard = bitboard & (bitboard >> 9)
        if(temp_bboard & (temp_bboard >> 2 * 9)):
            return True
        # Check |
        temp_bboard = bitboard & (bitboard >> 1)
        if(temp_bboard & (temp_bboard >> 2 * 1)):
            return True
        return False

    def move_won(self,i):
        return self.amove_won()

    def is_draw(self):
        return self.num_moves_made==42
    
    def reset(self):
        self.__init__()
        
    def data(self):
        array_rep = np.zeros((6,7),dtype=int32)
        for j in range(7):
            for i in range(6):
                array_rep[i,j] = ((self.p2>>(i +8*j))&1) - ((self.p1>>(i +8*j))&1)
        return array_rep[::-1]
    
    def show(self):
        plt.imshow(self.data())

class Connect4Game(object):
    def __init__(self,move_first=True,think_time=1):
        self.engine = MCTS(Connect4BitBoard)
        self.think_time = think_time
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
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(.1)
        
    def engine_move_update(self):
        engine_move =self.engine.compute_move(self.think_time)
        self.text_artist.set_text("{};{:1.2f}".format(self.engine.searchTree.num_visits,self.engine.searchTree.win_ratio()))
        outcome = self.engine.make_move(engine_move)
        if outcome: self.show_victory(outcome)
        self.ppt.set_data(self.engine.gameBoard.data())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
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