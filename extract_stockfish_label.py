import chess
import chess.uci
import chess.pgn
import sys
import numpy as np
import torch
import io


def sample_board(game):
    num_moves = len([0 for _ in game.mainline()])
    sampled_move_id = np.random.randint(num_moves-1)#exclude checkmates
    for i,g in enumerate(game.mainline()):
        sampled_game = g
        if i==sampled_move_id:
            break
    return sampled_game.board()

def pgn2labeledfen(pgn,engine,handler,evaltime=1e3):
    game = chess.pgn.read_game(io.StringIO(pgn))
    random_board = sample_board(game)
    board_fen = random_board.fen()
    engine.position(random_board)
    evaluation = engine.go(movetime=evaltime)
    bestmove = evaluation.bestmove.uci()
    boardscore = handler.info["score"][1].cp
    if boardscore is None:
        score = np.sign(handler.info['score'][1].mate)
    else: score = boardscore/100/25
    return board_fen,(score,bestmove)

class flatLabeler(object):
    def __init__(self,evaltime=1e3):
        self.handler = chess.uci.InfoHandler()
        self.engine = chess.uci.popen_engine(
            './stockfish-10-linux/Linux/stockfish_10_x64') 
        self.engine.info_handlers.append(self.handler)
        self.evaltime=evaltime
    def __call__(self,pgns):
        out = []
        for pgn in pgns:
            game = chess.pgn.read_game(io.StringIO(pgn))
            random_board = sample_board(game)
            board_fen = random_board.fen()
            self.engine.position(random_board)
            evaluation = self.engine.go(movetime=self.evaltime)
            bestmove = evaluation.bestmove.uci()
            boardscore = self.handler.info["score"][1].cp
            if boardscore is None:
                score = np.sign(self.handler.info['score'][1].mate)
            else: score = np.minimum(np.abs(boardscore)/100/8,1)*np.sign(boardscore)
            out.extend([(board_fen,(score,bestmove))])
        return out

pieces = 'PRNBQKprnbqk'
pieces2plane = {}
for i,piece_char in enumerate(pieces):
    pieces2plane[piece_char]=i
def encode(piece_char):
    return pieces2plane[piece_char]
def decode(index):
    return pieces[index]
    
def board2tensor(board):
    board_tensor = torch.zeros(12,8,8)
    for k,piece_char in board.piece_map().items():
        i,j = k//8,k%8
        board_tensor[encode(piece_char.symbol()),i,j] = 1
    return board_tensor

def fen2tensor(fen):
    board_tensor = torch.zeros(12,8,8)
    brd,color,castling,enps,hc,mvnum = fen.split(' ')
    for i,row in enumerate(brd.split('/')):
        j=0
        for char in row:
            if char.isdigit():
                j+=int(char)
            else:
                board_tensor[encode(char),i,j]=1
                j+=1
    return board_tensor



def collapse(board_tensor):
    summed = torch.sum(torch.arange(1,13,dtype=torch.float32)[:,None,None]*\
                        board_tensor,dim=0)
    return summed




# #give your position to the engine:
# engine.position(game.board())

# #Set your evaluation time, in ms:
# evaltime = 1000 #so 5 seconds
# evaluation = engine.go(movetime=evaltime)
# bestmove = evaluation.bestmove.uci()
# print(handler.info)
# boardscore = handler.info["score"][1].cp/100

if __name__=='__main__':
    pass
# #Now we have our board ready, load your engine:
# handler = chess.uci.InfoHandler()
# engine = chess.uci.popen_engine('./stockfish-10-linux/Linux/stockfish_10_x64') #give correct address of your engine here
# engine.info_handlers.append(handler)