import chess
import chess.uci
import chess.pgn
import sys

arguments = sys.argv
pgnfilename = str(arguments[1])

#Read pgn file:
with open(pgnfilename) as f:
    game = chess.pgn.read_game(f)

#Go to the end of the game and create a chess.Board() from it:
game = game.end()
board = game.board()

#So if you want, here's also your PGN to FEN conversion:
print('FEN of the last position of the game: ', board.fen())

#or if you want to loop over all game nodes:
#while not game.is_end():
    #node = game.variations[0]
    #board = game.board() #print the board if you want, to make sure
    #game = node         

#Now we have our board ready, load your engine:
handler = chess.uci.InfoHandler()
engine = chess.uci.popen_engine('./stockfish-10-linux/Linux/stockfish_10_x64') #give correct address of your engine here
engine.info_handlers.append(handler)

#give your position to the engine:
engine.position(board)

#Set your evaluation time, in ms:
evaltime = 5000 #so 5 seconds
evaluation = engine.go(movetime=evaltime)

#print best move, evaluation and mainline:
print('best move: ', board.san(evaluation[0]))
print('evaluation value: ', handler.info["score"][1].cp/100.0)
print('Corresponding line: ', board.variation_san(handler.info["pv"][1]))