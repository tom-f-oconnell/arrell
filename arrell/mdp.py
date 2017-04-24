# Tom O'Connell

import numpy as np
from . import util

class MDP:
    """
    S : set of states (implemented as vector of integers)
    A : set of actions (which i think should also just be a vector of integer labels for now)
    P : matrix of transition probabilities (A x S x S) -> [0,1]
        from Wiki: "...sometimes written as Pr(s,a,s'), Pr(s'|s,a)..."
        TODO which is correct?
    R : reward for taking action a in s, ending in s' (A x S x S) -> real
    discount : how import are past rewards? more -> closer to 1, less -> closer to zero.
    """
    def __init__(self, nS, nA, discount):
        assert 0 <= discount, 'discount must be non-negative'
        assert discount <= 1, 'value too large'

        self.S = np.arange(nS)
        # TODO are actions really necessary in MDPs? can you turn them in to states?
        self.A = np.arange(nA)
        self.discount = discount

        self.policy = self.initial_policy(self.S)
        self.values = self.initial_values(self.S)

        self.state = None


    def initial_policy(self, S):
        policy = dict()
        for s in S:
            actions = self.possible_actions(s)
            if len(actions) == 0:
                policy[s] = -1
            else:
                policy[s] = np.random.choice(actions)

        return policy


    def initial_values(self, S):
        values = dict()
        for s in S:
            values[s] = 0.0
        return values


    def possible_actions(self, s):
        return self.A

    
    def step(self, env):
        if self.state is None:
            raise ValueError('agent state was not initialized. do agent.state = s0.')
        a = self.policy[self.state]
        s, r, done, info = env.step(a)
        # TODO need update defined in this super class for it to work?
        self.update(a, s, r)
        self.state = s
        return s, r, done, info


class FullMDP(MDP):
    """
    Algorithms for finding policies for MDPs if you know state transition policies given actions
    and expected rewards.
    """
    def __init__(self, nS, nA, P, R, discount, learning_rate):
        #util.check_P(P)
        self.P = P
        self.R = R
        self.learning_rate = learning_rate
        super().__init__(nS, nA, discount)


    # TODO reasons i cant define this as 'a' index of P where P(a,s,:) has >= 1 nonzero?
    # maybe just inefficient in some cases
    def possible_actions(self, s):
        positive = np.argwhere(np.sum(self.P, axis=2).squeeze()).T
        return positive[0, positive[1,:] == s]


    # TODO fix
    def exact_value(self, actions, states, discount):
        total = 0
        # the agent has not taken an action yet in its current / terminal state
        # so len(actions) = len(states) - 1
        for t, a_curr, s_curr, s_next in zip(range(len(actions)), actions, states[:-1], states[1:]):
            # a_curr should be the entry from the policy, evaluated at time t from state s
            total = gamma**t * R(a_curr, s_curr, s_next)
        return total


    # TODO use matrix math
    def value(self, s, a):
        total = 0
        for s_prime in self.S:
            '''
            if s == 14 and s_prime == 15:
                print('action', a)
                print(self.P[a, s, s_prime] * \
                (self.R[a, s, s_prime] + self.discount * self.values[s_prime]))
            '''

            total += self.P[a, s, s_prime] * \
                (self.R[a, s, s_prime] + self.discount * self.values[s_prime])

        '''
        if s == 14:
            print('total:', total)
        '''
        return total


    # TODO maybe dont need to do this? if pi is used right before?
    def maxvalue(self, s):
        return max([self.value(s, a) for a in self.A])


    def pi(self, s):
        actions = self.possible_actions(s)
        '''
        if s == 14:
            print('possible actions', actions)
        '''
        best = -1
        best_value = None
        # for each action possible in current state
        for a in actions:
            curr_value = self.value(s, a)

            # TODO converge without equality as well?
            # random choice?
            if best_value == None or curr_value > best_value:
                best = a
                best_value = curr_value
        return best


    def compute_check_convergence(self, fn, store):
        """
        Returns whether policy changed (False if converged).
        """
        converged = True
        for s in self.S:
            new = fn(s)
            if store[s] != new:
                converged = False
            store[s] = new
        return converged


    # TODO might not want to check convergence on both. convergence in one probably 
    # implies convergence in both
    def update_policy(self):
        #print('updating policy...')
        return self.compute_check_convergence(self.pi, self.policy)


    def update_values(self):
        #print('updating values...')
        return self.compute_check_convergence(self.maxvalue, self.values)


    # TODO might not want to check convergence on both. convergence in one probably 
    # implies convergence in both (doesn't seem to, but i think policy update is broken)
    # TODO it seemed that these two steps are equivalent to single equation 
    # on Wiki under value iteration
    def value_iteration(self, n=None):
        i = 0
        #util.show_frozenlake(self.values)
        #util.show_frozenlake(self.policy)

        while n is None or i < n:
            pconverged = self.update_policy()
            vconverged = self.update_values()
            #util.show_frozenlake(self.values)
            #util.show_frozenlake(self.policy)

            for v in self.values.values():
                assert not np.isnan(v)

            i += 1
            #print(pconverged, vconverged)
            if pconverged and vconverged:
                print('policy converged after', i, 'iterations')
                break
        return i, self.policy, self.values


    def unprincipled_P_update(self, a, s):
        self.P[a, self.state, s] += self.learning_rate
        new_total = np.sum(self.P[a, self.state, :])
        for sp in self.S:
            self.P[a, self.state, sp] /= new_total
        assert np.isclose(np.sum(self.P[a, self.state, :]), 1.0)


    def unprincipled_R_update(self, a, s, r):
        self.R[a, self.state, s] = (1 - self.learning_rate) * self.R[a, self.state, s] + \
                self.learning_rate * r


    # TODO list available updates
    def update_P(a, s):
        raise ValueError('set update_P first as agent.update_P = <your choice of function>.' + \
                ' agent.unprincipled_P_update should work.')


    def update_R(a, s, r):
        raise ValueError('set update_R first as agent.update_R = <your choice of function>.' + \
                ' agent.unprincipled_R_update should work')
    

    def update(self, a, s, r):
        # TODO possible to do some kind of more principled P and R updates?
        # update models of transition probability and rewards
        #self.unprincipled_P_update(a, s)
        #self.unprincipled_R_update(a, s, r)
        self.update_P(a, s)
        self.update_R(a, s, r)

        # recalculate optimal policy
        self.value_iteration()

