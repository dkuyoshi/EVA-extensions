from chainer import Chain, Variable
from chainer import links as L
from chainer import functions as F

from value_buffer import ValueBuffer
from chainerrl.action_value import DiscreteActionValue
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.links import FactorizedNoisyLinear

class CNN(Chain):
    def __init__(self, n_history=4, n_hidden=256):
        super().__init__()
        self.n_hidden = n_hidden
        with self.init_scope():
            self.l1 = L.Convolution2D(n_history, 16, ksize=8, stride=4, nobias=False)
            self.l2 = L.Convolution2D(16, 32, ksize=4, stride=2, nobias=False)
            self.l3 = L.Convolution2D(32, 32, ksize=3, stride=1, nobias=False)
            self.l4 = L.Linear(1568, n_hidden)

    def __call__(self, x, test=False):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = self.l4(h)
        return h

class QNet(Chain):
    def __init__(self, n_history, num_actions, n_hidden=256):
        super().__init__()
        with self.init_scope():
            self.hout = CNN(n_history, n_hidden)
            self.qout = L.Linear(n_hidden, num_actions)

    def __call__(self, x, test=False):
        self.embedding = self.hout(x)
        return DiscreteActionValue(self.qout(F.relu(self.embedding)))

class QFunction(Chain):
    
    def __init__(self, n_history, num_actions, xp, LRU, n_hidden=256, lambdas=0.5, capacity=2000, num_neighbors=5):
        super().__init__()
        with self.init_scope():
            self.q_func = QNet(n_history, num_actions, n_hidden)
        # Call the initialization of the value buffer that outputs the non-parametric Q value
        self.non_q = ValueBuffer(capacity, num_neighbors, n_hidden, num_actions, xp=xp, LRU=LRU)

        self.lambdas = lambdas
        self.num_actions = num_actions
        self.capacity = capacity

    def __call__(self, x, eva=False, test=False):
        q = self.q_func(x)
        self.embedding = self.get_embedding()
        if not eva or self.lambdas==0 or self.lambdas==1:
            return q
        #Output Q-value from value buffer
        qnp = self.non_q.get_q(self.embedding.array)
        #Q-value adjustment
        qout = self.lambdas*q.q_values + (1-self.lambdas)*qnp
        return DiscreteActionValue(qout)

    def get_embedding(self):
        return self.q_func.embedding

class DuelingQNet(Chain):
    def __init__(self, n_history, num_actions, n_hidden=256):
        super().__init__()
        with self.init_scope():
            self.num_actions = num_actions
            self.hout = CNN(n_history, n_hidden)
            #self.fully_layer = L.Linear(n_hidden, 512)
            self.a_stream = L.Linear(n_hidden, num_actions)
            self.v_stream = L.Linear(n_hidden, 1)

    def __call__(self, x, test=False):
        self.embedding = self.hout(x)
        activation = F.relu(self.embedding)
        #activation = F.relu(self.fully_layer(l))
        batch_size = x.shape[0]
        ya = self.a_stream(activation)
        mean = F.reshape(F.sum(ya, axis=1) / self.num_actions, (batch_size, 1))
        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        ys = self.v_stream(activation)
        ya, ys = F.broadcast(ya, ys)
        q = ya + ys
        return DiscreteActionValue(q)

class DuelingQFunction(Chain):
    '''Use dueling network architecture
    '''
    def __init__(self, n_history, num_actions, xp, LRU, n_hidden=256, lambdas=0.5, capacity=2000, num_neighbors=5):
        super().__init__()
        with self.init_scope():
            self.q_func = DuelingQNet(n_history, num_actions, n_hidden)
        self.non_q = ValueBuffer(capacity, num_neighbors, n_hidden, num_actions, xp=xp, LRU=LRU)
        self.lambdas = lambdas
        self.num_actions = num_actions
        self.capacity = capacity

    def __call__(self, x, eva=False, test=False):
        q = self.q_func(x)
        self.embedding = self.get_embedding()
        if not eva or self.lambdas==0 or self.lambdas==1:
            return q
        qnp = self.non_q.get_q(self.embedding.array)
        qout = self.lambdas*q.q_values + (1-self.lambdas)*qnp
        return DiscreteActionValue(qout)

    def get_embedding(self):
        return self.q_func.embedding

