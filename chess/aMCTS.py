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

class NNevalQueue(queue.Queue):
    # batched_nn_executor holds a queue
    # when the queue is of length (batchsize) then the network
    # evaluates
    # may need its own thread (a daemon thread?)
    def __init__(self,network,batch_size=16):
        super().__init__()
        self.network = network
        self.outputTable = {}
        self.batch_size = batch_size
        self.worker = threading.Thread(target=self.work)
        self.worker.daemon = True
        self.worker.start()
        self.num_dispatches = 0

    def evaluate(self,board):
	#TODO: Switch keys to a random integer to prevent board collision
	# where multiple copies of the same board are submitted in the same
	# minibatch. Ideally avoiding locks & synchronization
        id = np.random.randint(2147483647)
        self.outputTable[id] = None
        self.put((id,board))
        while self.outputTable[id] is None:
            time.sleep(.0001)
        return self.outputTable.pop(id)

    def work(self):
        while True:
            if not self.empty():
                mb_boards = []
                ids = []
                for i in range(self.batch_size):
                    try: 
                        id,board = self.get(False)
                        mb_boards.append(board)
                        ids.append(id)
                    except queue.Empty: break
                #print("queue dispatch with {} boards".format(len(mb_boards)))
                encoded_mb = [self.network.encode(board) for board in mb_boards]
                collated_mb = default_collate(encoded_mb)
                values,logits = self.network(*collated_mb)
                values = values.data.cpu().numpy()
                moveProbs = F.softmax(logits,dim=1).data.cpu().numpy()
                for id, val, moveProb in zip(ids,values,moveProbs):
                    self.outputTable[id] = (val,moveProb)
                self.num_dispatches+=1
            else: time.sleep(.0001)
            
                
            
            
        
class SearchNode(object):
    C_PUCT = 2.5
    EPSILON = 1e-5
    def __init__(self, moveProbs):
        considered_mvs_mask = moveProbs > SearchNode.EPSILON
        self.mv_ids = np.arange(len(moveProbs))[considered_mvs_mask] # Move encodings
        self.Ps = np.ones(len(self.mv_ids))/len(self.mv_ids)#moveProbs[considered_mvs_mask] # Nonzero move probabilites
        self.Ns = np.zeros(len(self.mv_ids)) # Edge visit counts
        self.Vs = np.zeros(len(self.mv_ids)) # Edge values
        self.Qs = np.zeros(len(self.mv_ids))-1 # Vs/Ns (the mean Q value)
        self.children = [None]*len(self.mv_ids)

    @staticmethod #@lru_cache
    def newchild_and_value(board,transposition_table,eval_queue):
        # TODO: Check if board is in transposition table
        # if so return the cached results
        # output = transposition_table.get(board,None)
        # if output is None:
        if board.is_game_over():
            return None, board.outcome()
        value_abs, moveProbs = eval_queue.evaluate(board)
        child = SearchNode(moveProbs) #TODO: Deal with terminal board state, checkmate, draw, etc
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
            value = child.update_path(board,table,eval_queue)
        self.Vs[i] += value
        self.Qs[i] = self.Vs[i]/self.Ns[i]
        return value

    def select(self):
        """ Returns the move index according to PUCT"""
        sqrtSumN = np.sqrt(np.sum(self.Ns))
        Us = self.Ps*sqrtSumN/(1+self.Ns)
        mv_index = np.argmax(self.Qs + SearchNode.C_PUCT*Us)
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
                            self.board,self.trans_table,self.eval_queue)
    def make_action(self,move):
        mvindex = np.where(self.searchTree.mv_ids==self.board.nn_encode_move(move))[0][0]
        super().make_action(move)
        self.searchTree = self.searchTree.children[mvindex]
        if self.searchTree is None:
            self.searchTree,_ = SearchNode.newchild_and_value(
                        self.board,self.trans_table,self.eval_queue)

    def run_simulation(self):
        simulation_board = copy.deepcopy(self.board)
        self.searchTree.update_path(simulation_board,self.trans_table,self.eval_queue)
        return 0

    def run_k_simulations(self,k=100):
        for i in range(k):
            self.run_simulation()

    def think(self,thinktime=1):
        start_time = time.time()
        with futures.ThreadPoolExecutor(self.num_threads) as exc:
            active_simulations = {exc.submit(self.run_simulation) for i in range(self.num_threads)}
            while time.time() - start_time < thinktime:
                done, not_done = futures.wait(active_simulations,
                                              timeout=.02,return_when=futures.FIRST_COMPLETED)
                for future in done:
                    active_simulations.remove(future)
                    active_simulations.add(exc.submit(self.run_simulation))
            futures.wait(active_simulations)

    def compute_action(self):
        start_time =time.time()
        while time.time() - start_time < self.movetime:
            self.run_simulation()
        # with futures.ThreadPoolExecutor(self.num_threads) as exc:
        #     active_simulations = {exc.submit(self.run_simulation) for i in range(self.num_threads)}
        #     while time.time() - start_time < self.movetime:
        #         done, not_done = futures.wait(active_simulations,
        #                                       timeout=.02,return_when=futures.FIRST_COMPLETED)
        #         for future in done:
        #             active_simulations.remove(future)
        #             active_simulations.add(exc.submit(self.run_simulation))
        #     futures.wait(active_simulations)
        #mv_id = self.searchTree.mv_ids[np.argmax(self.searchTree.Ns)]
        #sqrtSumN = np.sqrt(np.sum(self.searchTree.Ns))
        #Us = self.searchTree.Ps*sqrtSumN/(1+self.searchTree.Ns)
        j=np.argmax(self.searchTree.Qs)
        mv_id = self.searchTree.mv_ids[j]
        action = self.board.nn_decode_move(mv_id)
        print("{} takes action {} with evals {}".format(['Black','White'][self.board.turn],action,self.searchTree.Ns[j]))
        print("# of Nodes evaluated: {}".format(np.sum(self.searchTree.Ns)))
        #j = np.argmax(self.searchTree.Ns)
        print("Yielding score: {}".format(self.searchTree.Vs[j]/(self.searchTree.Ns[j]+.00001)))
        print("Best scoring move was: _ with score {}".format(np.max(self.searchTree.Vs/(self.searchTree.Ns+.0001))))
        return action

    # def play_through_game()