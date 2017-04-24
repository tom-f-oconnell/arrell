# Tom O'Connell

import numpy as np
from . import mdp
from . import util

class QLearner(mdp.MDP):
    def __init__(self, nS, nA, discount, learning_rate=None):
        super().__init__(nS, nA, discount)
        self.learning_rate = learning_rate
        self.Q = np.zeros((nS, nA))

    def best_action(self, s):
        # noise following profile used in a medium.com tutorial post
        # not sure if there is some specific reason for it
        return np.argmax(self.Q[s, :] + np.random.randn(1, len(self.A)) * (1./(self.i / 1. + 1)))

    def update(self, a, s, r):
        """
        Called in the middle of mdp.MDP.step, which steps OpenAI gym environment
        and does a little extra bookkeeping.
        """
        # TODO should it update Q values for any states other than current one?
        self.Q[self.state, a] += self.learning_rate * (r + self.discount * \
                np.max(self.Q[s, :]) - self.Q[self.state, a])

        for s in self.S:
            self.policy[s] = self.best_action(s)

