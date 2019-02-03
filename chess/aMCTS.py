from agent import ChessBoard, Agent, NNAgent

from queue import Queue

class batched_nn_executor(object):
    # batched_nn_executor holds a queue
    # when the queue is of length (batchsize) then the network
    # evaluates
    # may need its own thread (a daemon thread?)
    def __init__(self,network):
        ...
        controlled = False
    def do_work_(self):
        values,moveProbs = self.network.infer(minibatch)
        # split values and moveProbs across the batch
        # filter out move probs below epsilon?
    def submit(self,board):
        self.q.put(..)
        if not controlled:
            controlled = True:
            while len(self.q) < 12:
                wait
                time.sleep(1e-6)
        if controlled:
            
            
        
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
    def newchild_and_value(board,transposition_table):
        # TODO: Check if board is in transposition table
        # if so return the cached results
        # output = transposition_table.get(board,None)
        # if output is None:
        value_abs, moveProbs = batched_NN_executor.submit(board)
        child = SearchNode(moveProbs)
        color = board.turn*2 - 1 # -1 or 1
        value_rel = value_abs*color
        output = (child, value_rel)
        #transposition_table[board] = output
        return output

    def update_path(self,board,table):
        i = self.select()
        # The virtual loss to prevent unwanted thread interaction
        # increments the visit counts before the value is propagated
        self.Ns[i] +=1 
        mv_id = self.mv_ids[i]
        board.make_action(board.nn_decode_move(mv_id))
        child = self.children[i]
        if child is None:
            self.children[i], value = self.newchild_and_value(board,table)
        else:
            value = child.update_path(board,table)
        self.Vs[i] += value
        #self.Qs[i] = self.Vs[i]/self.Ns[i]
        return -1*value

    def select(self):
        """ Returns the move index according to PUCT"""
        sqrtSumN = np.sqrt(np.sum(self.Ns))
        Us = SearchNode.C_PUCT*self.Ps*sqrtSumN/(1+self.Ns)
        Qs = self.Vs/self.Ns
        mv_index = np.argmax(Qs + Us)
        return mv_index


class MCTSAgent(NNAgent):
    def __init__(self,movetime=1,bs=64,num_threads=1):
        self.movetime=movetime
        self.searchTree = SearchNode.newchild_and_value(board)
        self.transposition_table = {}
    # def set_game_state(self,state):
    #     self.searchTree = SearchNode()
    # def make_action(self,move):
    #     pass
    def run_simulation(self,board):
        simulation_board = copy.copy(board)
        self.searchTree.update_path(simulation_board,self.transposition_table)

    def compute_action(self,move):
        start_time =time.time()
        with futures.ThreadPoolExecutor(self.num_threads) as exc:
            active_simulations = {exc.submit(self.run_simulation) for i in range(self.num_threads)}
            while time.time() - start_time < self.movetime:
                done, not_done = futures.wait(active_simulations,
                                              timeout=.02,return_when=futures.FIRST_COMPLETED)
                for future in done:
                    active_simulations.remove(future)
                    active_simulations.add(exc.submit(self.run_simulation))
        mv_id = self.searchTree.mv_ids[np.argmax(self.searchTree.Ns)]
        return self.board.nn_decode_move(mv_id)