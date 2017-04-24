# Tom O'Connell

import numpy as np
from . import mdp
from . import util

class QLearner(mdp.MDP):
    def __init__(self, nS, nA, discount, learning_rate=None):
        super().__init__(nS, nA, discount)
        self.learning_rate = learning_rate
        self.Q = np.zeros((nS, nA))


    def update(self, a, s, r):
        self.Q[self.state, a] += self.learning_rate * (r + self.discount * \
                np.max(self.Q[s, :]) - self.Q[self.state, a])

