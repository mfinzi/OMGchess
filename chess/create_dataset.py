import chess
import chess.uci
import chess.pgn
import sys,os
import numpy as np
import torch
import io
import pandas as pd
import concurrent
import dill
import argparse
import multiprocessing
import itertools
from oil.utils.mytqdm import tqdm


def sample_board(game):
    num_moves = len([0 for _ in game.mainline()])
    if num_moves <=1: raise StopIteration("Not enough moves in the pgn")
    sampled_move_id = np.random.randint(num_moves-1)#exclude checkmates
    for i,g in enumerate(game.mainline()):
        sampled_game = g
        if i==sampled_move_id:
            break
    return sampled_game.board()

def sample_kfen_history(game,k=4):
    num_moves = len([0 for _ in game.mainline()])
    if num_moves <=1: raise StopIteration("Not enough moves in the pgn")
    sampled_move_id = np.random.randint(num_moves-1)#exclude checkmates
    start_board_fen = chess.Board().fen()
    kmove_history = [start_board_fen]*k
    for i,g in enumerate(game.mainline()):
        if sampled_move_id -i <k:
            sampled_fen = g.board().fen()
            kmove_history = kmove_history[1:]+[sampled_fen]
        if i==sampled_move_id:
            break
    return kmove_history

class flatLabeler(object):
    def __init__(self,evaltime=1e3,k_history=4):
        self.handler = chess.uci.InfoHandler()
        self.engine = chess.uci.popen_engine(
            './stockfish-10-linux/Linux/stockfish_10_x64') 
        self.engine.info_handlers.append(self.handler)
        self.evaltime=evaltime
        self.k_history = k_history
    def __call__(self,pgns):
        out = []
        for pgn in pgns:
            try:
                game = chess.pgn.read_game(io.StringIO(pgn))
                try: random_history = sample_kfen_history(game,self.k_history)
                except StopIteration: continue
                random_board = chess.Board(random_history[-1])
                self.engine.position(random_board)
                evaluation = self.engine.go(movetime=self.evaltime)
                bestmove = evaluation.bestmove.uci()
                boardscore = self.handler.info["score"][1].cp
                if boardscore is None: # Todo increase score with closer mate
                    mate_info = self.handler.info['score'][1].mate
                    sign,turns = np.sign(mate_info),np.abs(mate_info)
                    score = sign*80*(1+1/(1+turns)**.5) # Encourages reducing the moves till mate
                else: score = boardscore
                if not random_board.turn:
                    score *= -1
                out.extend([(random_history,score,bestmove)])
            except Exception as e:
                print(e)
                continue
        return out

def create_labeled_dataset(pgnlist,num_workers=4,evaltime=1,num_jobs=None):
    num_elems = len(pgnlist)
    num_jobs = num_jobs or num_workers
    ratio = num_elems//num_jobs
    results = []
    with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
        ftrs = [executor.submit(flatLabeler(evaltime*1e3),pgnlist[i*ratio:(i+1)*ratio]) for i in range(num_jobs)]
        for future in tqdm(concurrent.futures.as_completed(ftrs),total=len(ftrs),desc='Dataset Pass'):
            #print("A worker completed its work and is shutting down")
            results+=future.result()
            yield results

def str2bool(k):
        if k=='true': return True
        if k=='false': return False
        return k

def read_pgns_with_annotations(filename):
    games=pd.read_table(filename,
                    sep='### ',
                    skiprows=[0,1,2,3,4],
                    names=['garbage','game'],
                    na_values='None',
                    engine='python',#nrows=nrows,
                    )['game']
    annotations=pd.read_table(filename,
                    sep=' ',usecols = np.arange(16),engine='c',
                skiprows=[0,1,2,3,4],
                    names=['t','date','result','welo','belo','len',
                        'date_c','resu_c','welo_c','belo_c','edate_c','setup','fen','resu2_c','oyrange','bad_len'],
                    na_values='None',#nrows=nrows,
                )
    annotations = annotations.apply(lambda st: pd.Series(str2bool(s.split('_')[-1]) if isinstance(s,str) else s for s in st ),axis=0)
    df = pd.concat([annotations, games], axis=1, sort=False)
    df = df[(df['setup'].values!=True)&(df['game'].values!=None)]
    return df

if __name__=='__main__':
    parser = argparse.ArgumentParser("Create chess dataset from annotated pgns")
    parser.add_argument('--train_size', type=int, default=3000000,
                    help='Number of games to label')
    parser.add_argument('-t','--time',metavar='time(s)',type=float,default=.1)
    parser.add_argument('--test_size', type=int,help='Size of the validation and test sets',default=10000)
    parser.add_argument('--positions',type=int,help='Positions to sample per game',default=10)
    parser.add_argument('--data_dir',type=str,help='Directory data will be saved to',default='data/')
    args = parser.parse_args()
    assert args.train_size<=3500000, "train_size exceeds dataset size"

    ncores = multiprocessing.cpu_count()-1
    print("Labeling {} game positions with stockfish using {}s per move and {} cores.".format(args.train_size,args.time,ncores))
    filename = os.path.join(args.data_dir,'all_with_filtered_anotations_since1998.txt')
    all_games = read_pgns_with_annotations(filename)['game']
    # Create train/val/test split and save the data indices
    indices = np.random.permutation(len(all_games))[:args.train_size+2*args.test_size]
    train_indices = indices[:args.train_size]
    val_indices = indices[args.train_size:args.train_size+args.test_size]
    test_indices = indices[args.train_size+args.test_size:args.train_size+2*args.test_size]
    with open(args.data_dir+"chess_{}k_{}s_indices.pkl".format(args.train_size//1000,args.time),'wb') as file:
        dill.dump((train_indices,val_indices,test_indices),file)
    # Create Val and Test Sets
    print("Creating Validation set of size {}\n".format(args.test_size))
    out_val = list(create_labeled_dataset(all_games.iloc[val_indices],ncores,args.time))[-1]
    with open(args.data_dir+"chess_{}k_{}s_val.pkl".format(args.train_size//1000,args.time),'wb') as file:
        dill.dump(out_val,file)

    print("Creating Test set of size {}\n".format(args.test_size))
    out_test = list(create_labeled_dataset(all_games.iloc[test_indices],ncores,args.time))[-1]
    with open(args.data_dir+"chess_{}k_{}s_test.pkl".format(args.train_size//1000,args.time),'wb') as file:
        dill.dump(out_test,file)

    print("Creating train_small set of size {}\n".format(args.test_size))
    out_train_small = list(create_labeled_dataset(all_games.iloc[train_indices][:args.test_size],ncores,args.time))[-1]
    with open(args.data_dir+"chess_{}k_{}s_trainsmall.pkl".format(args.train_size//1000,args.time),'wb') as file:
        dill.dump(out_train_small,file)
    # Create Train set saving as we go
    print("Creating train set of size {}\n".format(args.train_size))
    njobs = int(np.ceil(args.train_size*(args.time/10000))) # Give each process 300 minutes of work
    train_games = all_games.iloc[train_indices]
    
    for j in tqdm(range(args.positions),desc='All Passes'):
        print("Currently on dataset pass {}\n".format(j+1))
        out_train = create_labeled_dataset(train_games,ncores,args.time,njobs)
        for partial_train_set in out_train:
            with open(args.data_dir+"chess_{}k_{}s_train_{}.pkl".format(
                                    args.train_size//1000,args.time,j),'wb') as file:
                dill.dump(partial_train_set,file)
    
    
    
    