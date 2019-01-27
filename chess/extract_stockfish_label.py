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
                score = np.sign(self.handler.info['score'][1].mate)*20
            else: score = self.handler.info["score"][1].cp
            out.extend([(board_fen,score,bestmove)])
        return out


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