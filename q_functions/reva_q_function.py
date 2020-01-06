from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library

standard_library.install_aliases()  # NOQA

from builtins import *  # NOQA
from future import standard_library

standard_library.install_aliases()  # NOQA

from chainer import Chain, Variable
from chainer import links as L
from chainer import functions as F

from value_buffer import ValueBuffer
from chainerrl.action_value import DiscreteActionValue, DistributionalDiscreteActionValue
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl.links import FactorizedNoisyLinear
from chainerrl import action_value
from chainerrl.links.mlp import MLP
import numpy as np


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


class NoisyQNet(Chain):
    def __init__(self, n_history, num_actions, n_atoms, v_min, v_max, n_hidden=256, sigma=0.5):
        super().__init__()
        with self.init_scope():
            self.hout = CNN(n_history, n_hidden)
            self.qout = FactorizedNoisyLinear(L.Linear(n_hidden, num_actions * n_atoms) ,sigma)
        self.n_atoms = n_atoms
        self.num_actions = num_actions
        z_values = self.xp.linspace(v_min, v_max, num=n_atoms, dtype=np.float32)
        self.add_persistent('z_values', z_values)

    def __call__(self, x, test=False):
        self.embedding = self.hout(x)
        q = self.qout(F.relu(self.embedding))
        batch_size = x.shape[0]
        q = F.reshape(q, (batch_size, self.num_actions, self.n_atoms))
        q = F.softmax(q, axis=2)
        return DistributionalDiscreteActionValue(q, self.z_values)

class NoisyDistributionalQFunction(Chain):

    def __init__(self, n_history, num_actions, xp, LRU, n_atoms, v_min, v_max, n_hidden=256,
                 lambdas=0.5, capacity=2000, num_neighbors=5, sigma=0.5):
        super().__init__()
        with self.init_scope():
            self.q_func = NoisyQNet(n_history, num_actions, n_atoms, v_min, v_max, n_hidden, sigma)
        self.non_q = ValueBuffer(capacity, num_neighbors, n_hidden, num_actions, xp=xp, LRU=LRU)

        self.lambdas = lambdas
        self.num_actions = num_actions
        self.capacity = capacity
        self.n_atoms = n_atoms

    def __call__(self, x, eva=False, test=False):
        """TODO: stateを受け取って, Q値を返す"""
        q = self.q_func(x)
        self.embedding = self.get_embedding()
        if not eva or self.lambdas == 0 or self.lambdas == 1:
            return q

        qnp = self.non_q.get_q(self.embedding.array)
        qout = self.lambdas * q.q_values + (1 - self.lambdas) * qnp
        return DiscreteActionValue(qout)

    def get_embedding(self):
        return self.q_func.embedding


class NoisyDuelingQNet(Chain):
    def __init__(self, n_history, num_actions, n_atoms, v_min, v_max ,n_hidden=256, sigma=0.5):
        super().__init__()
        with self.init_scope():
            self.hout = CNN(n_history, n_hidden)
            # self.fully_layer = L.Linear(n_hidden, 512)
            self.a_stream = FactorizedNoisyLinear(L.Linear(n_hidden, num_actions * n_atoms),sigma)
            self.v_stream = FactorizedNoisyLinear(L.Linear(n_hidden, n_atoms), sigma)

        self.num_actions = num_actions
        self.n_atoms = n_atoms

        z_values = self.xp.linspace(v_min, v_max, num=n_atoms, dtype=np.float32)
        self.add_persistent('z_values', z_values)

    def __call__(self, x, test=False):
        self.embedding = self.hout(x)
        activation = F.relu(self.embedding)
        # activation = F.relu(self.fully_layer(l))
        # h_a, h_v = F.split_axis(activation, 2, axis=-1)
        batch_size = x.shape[0]
        ya = F.reshape(self.a_stream(activation), (batch_size, self.num_actions, self.n_atoms))
        mean = F.sum(ya, axis=1, keepdims=True) / self.num_actions
        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        ys = F.reshape(self.v_stream(activation), (batch_size, 1, self.n_atoms))
        ya, ys = F.broadcast(ya, ys)
        q = F.softmax(ya + ys, axis=2)
        return DistributionalDiscreteActionValue(q, self.z_values)


class NoisyDistributonalDuelingQFunction(Chain):
    def __init__(self, n_history, num_actions, xp, LRU, n_atoms, v_min, v_max, n_hidden=256,
                 lambdas=0.5, capacity=2000, num_neighbors=5, sigma=0.5):
        super().__init__()
        with self.init_scope():
            self.q_func = NoisyDuelingQNet(n_history, num_actions, n_atoms, v_min, v_max, n_hidden, sigma)

        self.non_q = ValueBuffer(capacity, num_neighbors, n_hidden, num_actions, xp=xp, LRU=LRU)

        self.lambdas = lambdas
        self.num_actions = num_actions
        self.capacity = capacity
        self.n_atoms = n_atoms


    def __call__(self, x, eva=False, test=False):
        q = self.q_func(x)
        self.embedding = self.get_embedding()
        if not eva or self.lambdas == 0 or self.lambdas == 1:
            return q

        qnp = self.non_q.get_q(self.embedding.array)
        qout = self.lambdas * q.q_values + (1 - self.lambdas) * qnp
        return DiscreteActionValue(qout)

    def get_embedding(self):
        return self.q_func.embedding