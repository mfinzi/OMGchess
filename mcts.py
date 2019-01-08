import numpy as np
from numba import jit,njit,int32,float32,jitclass,void,boolean
import time
import math
from math import sqrt


class Node(object):
    
    # num_rollouts = 0
    # sqrtlog_num_rollouts = 0
    # temperature = 1
    # transposition_table = {}
    # reused=0
    
    @classmethod
    def reset(cls):
        cls.num_rollouts = 0
        cls.sqrtlog_num_rollouts = 0
        cls.temperature = .5
        cls.transposition_table = {}
        cls.reused=0
        
    __slots__ = ('move','children','unvisited','num_visits','num_wins','sqrtlogN')
    def __init__(self,move): #move, number of visits, wins
        self.move = move
        self.children = []
        self.unvisited = None
        self.num_visits = 0
        self.num_wins = 0
        self.sqrtlogN = 0
    
    @staticmethod
    @jit(nopython=True)
    def rollout(board):
        while True:
            moves = board.get_moves()
            if len(moves)==0: return 0 # a draw
            move = moves[np.random.randint(len(moves))]
            outcome = board.make_move(move)
            if outcome: return outcome
            
    def win_ratio(self):
        win_rate = self.num_wins/self.num_visits
        #explore_bonus = Node.sqrtlog_num_rollouts/math.sqrt(self.num_visits)
        return win_rate# - explore_bonus
    
    def best_child(self,final=False):
        # fix final
        key = lambda child: self.ucb(child.num_wins,child.num_visits,self.sqrtlogN)
        return max(self.children,key=key)

    @staticmethod
    @jit(nopython=True)
    def ucb(num_wins,num_visits,sqrtlogN):
        win_rate = num_wins/num_visits
        explore_bonus = sqrtlogN/math.sqrt(num_visits)
        return win_rate + explore_bonus

    def terminal_outcome(self,board):
        color = board.color_to_move()
        won = board.move_won(self.move)
        if won: return won*color*(-1) # victory
        if board.is_draw(): return 0 # draw
        return None # Nothing
    
    def update_path(self,board):
        color = board.color_to_move()
        
        # leaf node, either terminal or unvisited
        if len(self.children)==0:
            terminal_outcome = self.terminal_outcome(board)
            if terminal_outcome is not None: outcome = terminal_outcome
            else:
                self.children = [Node(k) for k in board.get_moves()]
                self.unvisited = np.random.permutation(len(self.children))
                outcome = self.rollout(board)
                #Node.num_rollouts +=1
                #Node.sqrtlog_num_rollouts = Node.temperature*np.sqrt(2*np.log(Node.num_rollouts))
                
        # Node has not been fully expanded
        elif len(self.unvisited):#np.any(self.unvisited):
            child = self.expand_unvisited(board)
            outcome = child.update_path(board)
            
        # Node has been fully expanded and we use the (ucb) policy    
        else:
            child = self.best_child()
            board.make_move(child.move)
            outcome = child.update_path(board)
            
        self.update_statistics(color,outcome)
        return outcome
    
    def update_statistics(self,color,outcome):
        self.num_visits +=1
        self.num_wins += 0.5*(1-color*outcome)
        self.sqrtlogN = np.sqrt(np.log(self.num_visits))
        

    def expand_unvisited(self,board):
        """ Finds an unvisited child node, adds/checks transpose table,
         makes move, returns child"""
        m = self.unvisited[-1]
        self.unvisited = self.unvisited[:-1]
        child = self.children[m]
        board.make_move(child.move)
#         key = hashkey(board)
#         if key in Node.transposition_table:
#             Node.reused+=1
#             self.children[m] = Node.transposition_table[key]
#             child = self.children[m]
#         else:
#             Node.transposition_table[key]=child
        return child

class MCTS(object):
    def __init__(self,boardType):
        self.boardType = boardType
        self.gameBoard = boardType()
        self.searchTree = Node(0)
        self.interrupt=False
        Node.reset()
        
    def ponder(self,think_time=np.inf):
        self.interrupt=False
        start_time = time.time()
        new_board = self.boardType()
        while time.time() - start_time < think_time and not self.interrupt:
            new_board.copy(self.gameBoard)
            self.searchTree.update_path(new_board)
    
    def compute_move(self,think_time):
#         start_time = time.time()
#         new_board = Connect4Board()
#         while time.time() - start_time < think_time:
#             new_board.copy(self.gameBoard)
#             self.searchTree.update_path(new_board)
        self.ponder(think_time)
        return self.searchTree.best_child(True).move
    
    def make_move(self,move):
        legal_moves = np.array(self.gameBoard.get_moves())
        assert move in legal_moves
        # find the associated child
        i = np.nonzero(move==legal_moves)[0][0]
        if self.searchTree.children:
            child = self.searchTree.children[i]
        else:
            child = Node(move)
        # Update the search tree
        # Discard the tree and table for now
        self.searchTree = child
        Node.num_rollouts = self.searchTree.num_visits
        #Node.reset()
        #self.searchTree = Node(move)
        outcome = self.gameBoard.make_move(self.searchTree.move)
        return outcome

