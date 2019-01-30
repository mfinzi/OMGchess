
import chess
import chess.svg
from chess_dataset import fen2tensor,legal_board_moves,move2class,class2move
from IPython.display import display


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
    def nn_decode_move(self,classid):
        #TODO: handle promotions
        return class2move(classid)
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
    def set_game_state(self,state):
        self.board.set_state(state)
    def make_action(self,move):
        self.board.make_move(move)
    def compute_action(self,time):
        raise NotImplementedError

class NNAgent(Agent):
    def __init__(self,GameType,network):
        super().__init__(GameType)
        self.network = network
        
    def compute_action(self,time):
        nn_board = self.board.nn_encode_board().unsqueeze(0).cuda()
        nn_legal_mvs = self.board.nn_legal_moves().unsqueeze(0).cuda()
        values,logits = self.network(nn_board,nn_legal_mvs)
        chosen_classid = logits.max(1)[1].squeeze().cpu().numpy()
        move = self.board.nn_decode_move(chosen_classid)
        return move

class KeyBoardAgent(Agent):
    def compute_action(self,time):
        move = input("Human move:")
        if not self.board.move_is_legal(move):
            move = input("{} is not a legal move. Try again.".format(move))
        return move

# class Game(object):
#     def __init__(self,agent1,agent2):
#         self.agent1 = agent1
#         self.agent2 = agent2
#         self.board = self.agent1.board # assume the same as agent2 board

class ChessGame(object):
    def __init__(self,agent1,agent2,time=1,display=True):
        self.agent1 = agent1
        self.agent2 = agent2
        self.time = time
        self.display = display
        self.play()
    def play(self):
        i=0
        board = self.agent1.board
        while not board.is_game_over():
            if self.display: display(board)
            player_up = [self.agent1,self.agent2][i%2]
            move = player_up.compute_action(self.time)
            self.agent1.make_action(move)
            self.agent2.make_action(move)
            i+=1
        return None # return game outcome
            




class MCTSAgent(Agent):
    pass