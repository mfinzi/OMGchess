from agent import ChessBoard, Agent, NNAgent
import threading
import queue
from torch.utils.data.dataloader import default_collate
import torch
import torch.nn.functional as F
import numpy as np
from concurrent import futures
import time
import copy
# class QueingNNworker(threading.Thread):
#     # batched_nn_executor holds a queue
#     # when the queue is of length (batchsize) then the network
#     # evaluates
#     # may need its own thread (a daemon thread?)
#     def __init__(self,network,*args,**kwargs):
#         self.network = network
#         self.queue = queue.Queue()
#         self.outputTable = {}
#         super().__init__(*args,**kwargs)

#     def run(self):
#         while True:
#             try: board = self.queue.get(timeout=1)
#         values,moveProbs = self.network.infer(minibatch)
#         # split values and moveProbs across the batch
#         # filter out move probs below epsilon?
#     def enqueue(self,board):
#         self.outputTable[board] = None
#         self.queue.put(board)

class NNevalQueue(queue.Queue):
    # batched_nn_executor holds a queue
    # when the queue is of length (batchsize) then the network
    # evaluates
    # may need its own thread (a daemon thread?)
    def __init__(self,network,batch_size=16):
        super().__init__()
        self.network = network
        self.force_eval = False
        self.outputTable = {}
        self.batch_size = batch_size
        self.worker = threading.Thread(target=self.work)
        self.worker.daemon = True
        self.worker.start()

    def enqueue(self,board,force_eval=False):
        self.outputTable[board] = None
        self.put(board)
        self.force_eval=force_eval

    def work(self):
        mb_boards = []
        while True:
            if len(mb_boards)>=self.batch_size or (self.force_eval and len(mb_boards)):
                encoded_mb = [self.network.encode(board) for board in mb_boards]
                collated_mb = default_collate(encoded_mb)
                values,logits = self.network(*collated_mb)
                values = values.data.cpu().numpy()
                moveProbs = F.softmax(logits,dim=1).data.cpu().numpy()
                for board, val, moveProb in zip(mb_boards,values,moveProbs):
                    self.outputTable[board] = (val,moveProb)
                self.force_eval = False
            else:
                try:
                    mb_boards.append(self.get(False))
                    self.task_done()
                except queue.Empty:
                    pass
            time.sleep(.0001)
            
                
            
            
        
class SearchNode(object):
    C_PUCT = 2.5
    EPSILON = 1e-5
    def __init__(self, moveProbs):
        considered_mvs_mask = moveProbs > SearchNode.EPSILON
        self.mv_ids = np.arange(len(moveProbs))[considered_mvs_mask] # Move encodings
        self.Ps = moveProbs[considered_mvs_mask] # Nonzero move probabilites
        self.Ns = np.zeros(len(self.mv_ids)) # Edge visit counts
        self.Vs = np.zeros(len(self.mv_ids)) # Edge values
        #self.Qs = np.zeros(len(self.mv_ids)) # Vs/Ns (the mean Q value)
        self.children = [None]*len(self.mv_ids)

    @staticmethod #@lru_cache
    def newchild_and_value(board,transposition_table,eval_queue,force=False):
        # TODO: Check if board is in transposition table
        # if so return the cached results
        # output = transposition_table.get(board,None)
        # if output is None:
        eval_queue.enqueue(board,force_eval=force)
        while eval_queue.outputTable[board] is None:
            time.sleep(.0001)
        value_abs, moveProbs = eval_queue.outputTable.pop(board)
        child = SearchNode(moveProbs)
        color = board.turn*2 - 1 # -1 or 1
        value_rel = value_abs*color
        output = (child, value_rel)
        #transposition_table[board] = output
        return output

    def update_path(self,board,table,eval_queue):
        i = self.select()
        # The virtual loss to prevent unwanted thread interaction
        # increments the visit counts before the value is propagated
        self.Ns[i] +=1 
        mv_id = self.mv_ids[i]
        board.make_move(board.nn_decode_move(mv_id))
        child = self.children[i]
        if child is None:
            self.children[i], value = self.newchild_and_value(board,table,eval_queue)
        else:
            value = child.update_path(board,table)
        self.Vs[i] += value
        #self.Qs[i] = self.Vs[i]/self.Ns[i]
        return -1*value

    def select(self):
        """ Returns the move index according to PUCT"""
        sqrtSumN = np.sqrt(np.sum(self.Ns))
        Us = self.Ps*sqrtSumN/(1+self.Ns)
        Qs = self.Vs/(self.Ns+.1)
        mv_index = np.argmax(Qs + SearchNode.C_PUCT*Us)
        return mv_index


class MCTSAgent(Agent):
    def __init__(self,GameType,network,movetime=1,bs=12,num_threads=30):
        super().__init__(GameType)
        self.num_threads = num_threads
        self.movetime=movetime
        self.trans_table = {}
        network.eval()
        self.eval_queue = NNevalQueue(network,batch_size=bs)
        self.searchTree,_ = SearchNode.newchild_and_value(
                            self.board,self.trans_table,self.eval_queue,force=True)
    def make_action(self,move):
        mvindex = np.where(self.searchTree.mv_ids==self.board.nn_encode_move(move))[0][0]
        self.searchTree = self.searchTree.children[mvindex]
        super().make_action(move)

    def run_simulation(self):
        simulation_board = copy.deepcopy(self.board)
        self.searchTree.update_path(simulation_board,self.trans_table,self.eval_queue)
        return 0

    def compute_action(self):
        start_time =time.time()
        with futures.ThreadPoolExecutor(self.num_threads) as exc:
            active_simulations = {exc.submit(self.run_simulation) for i in range(self.num_threads)}
            while time.time() - start_time < self.movetime:
                done, not_done = futures.wait(active_simulations,
                                              timeout=.2,return_when=futures.FIRST_COMPLETED)
                for future in done:
                    active_simulations.remove(future)
                    active_simulations.add(exc.submit(self.run_simulation))
            futures.wait(active_simulations)
        mv_id = self.searchTree.mv_ids[np.argmax(self.searchTree.Ns)]
        action = self.board.nn_decode_move(mv_id)
        print(action)
        print(np.sum(self.searchTree.Ns))
        return action