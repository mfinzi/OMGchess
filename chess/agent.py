
import chess
import chess.svg
import chess.engine
import chess.pgn
from chess_dataset import fen2tensor,legal_board_moves,move2class,class2move,legal_opponent_moves
from IPython.display import display
from oil.utils.utils import Named
import tempfile
import atexit,os
import numpy as np

class ChessBoard(chess.Board):
    #reset
    def state(self):
        return self.fen()
    def set_state(self,fen):
        self.set_fen(fen)
    def nn_encode_board(self):
        return fen2tensor(self.fen())
    def nn_legal_moves(self):
        return legal_board_moves(self)
    def nn_opp_moves(self):
        self.turn = not self.turn
        opp_moves = legal_board_moves(self)
        self.turn = not self.turn
        return opp_moves
    def nn_decode_move(self,classid):
        #TODO: handle promotions
        movestr = class2move(classid)
        m = chess.Move.from_uci(movestr)
        if chess.square_rank(m.to_square) in (0,7) and \
            (self.piece_at(m.from_square).symbol() in ('P','p')):
            movestr += 'q' # Queen promotions only
        return movestr
    def make_move(self,uci_str):
        move = chess.Move.from_uci(uci_str)
        self.push(move)
    def move_is_legal(self,uci_str):
        if len(uci_str) not in (4,5): return False
        return chess.Move.from_uci(uci_str) in self.legal_moves
    def unmake_move(self):
        self.pop()
    def key(self):
        raise NotImplementedError
    def as_svg(self):
        return chess.svg.board(self)

class Agent(object):
    def __init__(self,GameType):
        self.board = GameType()
    def reset(self):
        # Will need to reset time controls here too
        self.board.reset()
    def set_game_state(self,state):
        self.board.set_state(state)
    def make_action(self,move):
        self.board.make_move(move)
    def compute_action(self):
        raise NotImplementedError
    def __str__(self):
        return self.__class__.__name__

class NNAgent(Agent):
    def __init__(self,GameType,network):
        super().__init__(GameType)
        self.network = network
        
    def compute_action(self):
        nn_board = self.board.nn_encode_board().unsqueeze(0).cuda()
        nn_legal_mvs = self.board.nn_legal_moves().unsqueeze(0).cuda()
        #nn_opp_mvs = self.board.nn_opp_moves().unsqueeze(0).cuda()
        values,logits = self.network(nn_board,nn_legal_mvs)#,nn_opp_mvs)
        chosen_classid = logits.max(1)[1].squeeze().cpu().numpy()
        move = self.board.nn_decode_move(chosen_classid)
        return move

    def __str__(self):
        return super().__str__() + '({})'.format(self.network.__class__.__name__)

class KeyBoardAgent(Agent):
    def compute_action(self):
        move = input("Human move:")
        if not self.board.move_is_legal(move):
            move = input("{} is not a legal move. Try again.".format(move))
        return move

class StockFishAgent(Agent):
    def __init__(self,thinktime=1,
                stockfish_path = './stockfish-10-linux/Linux/stockfish_10_x64'):
        self.thinktime = thinktime
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        super().__init__(ChessBoard)

    def compute_action(self):
        result = self.engine.play(self.board,chess.engine.Limit(time=self.thinktime))
        return result.move.uci()

    def __str__(self):
        return super().__str__() + '({}s)'.format(self.thinktime)
# class Game(object):
#     def __init__(self,agent1,agent2):
#         self.agent1 = agent1
#         self.agent2 = agent2
#         self.board = self.agent1.board # assume the same as agent2 board
# TODO: Time controls (stockfish has built in ability to alot time)
class ChessGame(object):
    def __init__(self,agent1,agent2,display=True):
        self.agent1 = agent1
        self.agent2 = agent2
        self.display = display

    def play(self):
        i=0
        self.agent1.reset()
        self.agent2.reset()
        board = self.agent1.board
        while not board.is_game_over():
            if self.display: display(board)
            player_up = [self.agent1,self.agent2][i%2]
            move = player_up.compute_action()
            self.agent1.make_action(move)
            self.agent2.make_action(move)
            i+=1
        return board # return game outcome

def getRelativeScore(agent1,agent2,num_games=30):
    """ Returns the mean score for agent1 and its standard dev"""
    scoring = {'0-1':0,'1/2-1/2':1/2,'1-0':1}
    outcomes = []
    # Half as white
    game = ChessGame(agent1,agent2,display=False)
    for i in range(num_games//2):
        outcomes += [scoring[game.play().result()]]
    # Half as black
    game = ChessGame(agent2,agent1,display=False)
    for i in range((num_games+1)//2):
        outcomes += [scoring[game.play().result()]]
    return np.mean(outcomes), np.std(outcomes)/np.sqrt(num_games)

class MCTSAgent(Agent):
    pass