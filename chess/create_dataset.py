import chess
import chess.uci
import chess.pgn
import sys
import numpy as np
import torch
import io
import pandas as pd
import concurrent
import dill
import argparse
import multiprocessing
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

class flatLabeler(object):
    def __init__(self,evaltime=1e3):
        self.handler = chess.uci.InfoHandler()
        self.engine = chess.uci.popen_engine(
            './stockfish-10-linux/Linux/stockfish_10_x64') 
        self.engine.info_handlers.append(self.handler)
        self.evaltime=evaltime
    def __call__(self,pgns):
        out = []
        for pgn in tqdm(pgns):
            try:
                game = chess.pgn.read_game(io.StringIO(pgn))
                try: random_board = sample_board(game)
                except StopIteration: continue
                board_fen = random_board.fen()
                self.engine.position(random_board)
                evaluation = self.engine.go(movetime=self.evaltime)
                bestmove = evaluation.bestmove.uci()
                boardscore = self.handler.info["score"][1].cp
                if boardscore is None:
                    score = np.sign(self.handler.info['score'][1].mate)*20
                else: score = self.handler.info["score"][1].cp
                out.extend([(board_fen,score,bestmove)])
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
        for future in tqdm(concurrent.futures.as_completed(ftrs),total=len(ftrs),desc='Total_work'):
            #print("A worker completed its work and is shutting down")
            results+=future.result()
            yield results

def str2bool(k):
        if k=='true': return True
        if k=='false': return False
        return k

def read_pgns_with_annotations():
    games=pd.read_table('all_with_filtered_anotations_since1998.txt',
                    sep='### ',
                    skiprows=[0,1,2,3,4],
                    names=['garbage','game'],
                    na_values='None',
                    engine='python',#nrows=nrows,
                    )['game']
    annotations=pd.read_table('all_with_filtered_anotations_since1998.txt',
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
    parser.add_argument('train_size', metavar='numrows', type=int,
                    help='Number of games to label')
    parser.add_argument('-t','--time',metavar='time(s)',type=float,default=5)
    parser.add_argument('--test_size', type=int,help='Size of the validation and test sets',default=1000)
    args = parser.parse_args()
    assert args.train_size<=3500000, "train_size exceeds dataset size"

    ncores = multiprocessing.cpu_count()-1
    print("Labeling {} game positions with stockfish using {}s per move and {} cores.".format(args.train_size,args.time,ncores))
    all_games = read_pgns_with_annotations()['game']
    indices = np.random.permutation(len(all_games))[:args.train_size+2*args.test_size]
    train_indices = indices[:args.train_size]
    val_indices = indices[args.train_size:args.train_size+args.test_size]
    test_indices = indices[args.train_size+args.test_size:args.train_size+2*args.test_size]

    # Create Val and Test Sets
    print("Creating Validation set of size {}\n".format(args.test_size))
    out_val = list(create_labeled_dataset(all_games.iloc[val_indices],ncores,args.time))[-1]
    with open("chess_{}k_{}s_val.pkl".format(args.train_size//1000,args.time),'wb') as file:
        dill.dump(out_val,file)
    print("Creating Test set of size {}\n".format(args.test_size))
    out_test = list(create_labeled_dataset(all_games.iloc[test_indices],ncores,args.time))[-1]
    with open("chess_{}k_{}s_test.pkl".format(args.train_size//1000,args.time),'wb') as file:
        dill.dump(out_test,file)
    print("Creating train_small set of size {}\n".format(args.test_size))
    out_train_small = list(create_labeled_dataset(all_games.iloc[train_indices][:1000],ncores,args.time))[-1]
    with open("chess_{}k_{}s_trainsmall.pkl".format(args.train_size//1000,args.time),'wb') as file:
        dill.dump(out_train_small,file)
    # Create Train set saving as we go
    print("Creating train set of size {}\n".format(args.train_size))
    njobs = int(np.ceil(args.train_size*(args.time/2000))) # Give each process 30 minutes of work
    out_train = create_labeled_dataset(all_games.iloc[train_indices],ncores,args.time,njobs)
    with open("chess_{}k_{}s_train.pkl".format(args.train_size//1000,args.time),'wb') as file:
        for partial_train_set in out_train:
            dill.dump(partial_train_set,file)
    
    
    
    