from chainer import Chain, Variable
from chainer import links as L
from chainer import functions as F

from value_buffer import ValueBuffer
from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.links import FactorizedNoisyLinear

class NN(Chain):
    def __init__(self, obs_size, n_hidden=32, n_out=8):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_out = n_out
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden)
            self.l1 = L.Linear(n_hidden, n_out)

    def __call__(self, x, test=False):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return h

class QNetCart(Chain):
    def __init__(self, obs_size, num_actions, n_hidden=8):
        super().__init__()
        with self.init_scope():
            self.hout = NN(obs_size)
            self.qout = L.Linear(n_hidden, num_actions)

    def __call__(self, x, test=False):
        self.embedding = self.hout(x)
        return DiscreteActionValue(self.qout(F.relu(self.embedding)))

class QFunctionCart(Chain):
    def __init__(self, obs_size, num_actions, xp, LRU, n_hidden=8, lambdas=0.5, capacity=2000, num_neighbors=5):
        super().__init__()
        with self.init_scope():
            self.q_func = QNetCart(obs_size, num_actions, n_hidden)
        # Call the initialization of the value buffer that outputs the non-parametric Q value
        self.non_q = ValueBuffer(capacity, num_neighbors, n_hidden, num_actions, xp=xp, LRU=LRU)
        self.capacity = capacity
        self.lambdas = lambdas
        self.num_actions = num_actions

    def __call__(self, x, eva=False, test=False):
        """TODO: stateを受け取って, Q値を返す"""
        q = self.q_func(x)
        self.embedding = self.get_embedding()
        if not eva or self.lambdas == 0:
            return q
        # Output the non-Q from value buffer
        qnp = self.non_q.get_q(self.embedding.array)
        #Q-value adjustment
        qout = self.lambdas * q.q_values + (1 - self.lambdas) * qnp
        return DiscreteActionValue(qout)

    def get_embedding(self):
        """return the embedding"""
        return self.q_func.embedding