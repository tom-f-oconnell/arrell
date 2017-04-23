# Tom O'Connell

import numpy as np

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
    def __init__(self, nS, nA, P, R, discount):
        assert 0 <= discount, 'discount must be non-negative'
        #assert discount < 1, 'value would runaway in infinite time'

        self.S = np.arange(nS)
        # TODO are actions really necessary in MDPs? can you turn them in to states?
        self.A = np.arange(nA)
        self.P = P
        self.R = R

        self.check_P()
        
        self.discount = discount

        self.policy = self.initial_policy(self.S)
        self.values = self.initial_values(self.S)


    def initial_policy(self, S):
        policy = dict()
        for s in S:
            policy[s] = np.random.choice(self.possible_actions(s))
        return policy


    def initial_values(self, S):
        values = dict()
        for s in S:
            values[s] = 0
        return values


    # TODO fix
    def exact_value(self, actions, states, discount):
        total = 0
        # the agent has not taken an action yet in its current / terminal state
        # so len(actions) = len(states) - 1
        for t, a_curr, s_curr, s_next in zip(range(len(actions)), actions, states[:-1], states[1:]):
            # a_curr should be the entry from the policy, evaluated at time t from state s
            total = gamma**t * R(a_curr, s_curr, s_next)
        return total


    # TODO reasons i cant define this as 'a' index of P where P(a,s,:) has >= 1 nonzero?
    # maybe just inefficient in some cases
    def possible_actions(self, s):
        positive = np.argwhere(np.sum(self.P, axis=2).squeeze()).T
        return positive[0, positive[1,:] == s]


    # TODO use matrix math
    def value(self, s, a):
        a = self.policy[s]
        total = 0
        for s_prime in self.S:
            '''
            print('s_prime', s_prime)
            print('p entry', self.P[a, s, s_prime])
            print('r entry', self.R[a, s, s_prime])
            print('P', self.P)
            print('R', self.R)
            '''
            total += self.P[a, s, s_prime] * \
                (self.R[a, s, s_prime] + self.discount * self.values[s_prime])
        return total


    def maxvalue(self, s):
        return max([self.value(s, a) for a in self.A])


    def pi(self, s):
        actions = self.possible_actions(s)
        best = None
        best_value = None
        # for each action possible in current state
        for a in actions:
            curr_value = self.value(s, a)

            # TODO converge without equality as well?
            if best_value == None or curr_value > best_value:
                best = a
                best_value = curr_value
        return a


    def compute_check_convergence(self, fn, store):
        """
        Returns whether policy changed (False if converged).
        """
        converged = True
        for s in self.S:
            new = fn(s)
            '''
            print('state', s)
            print('new', new)
            print('old', store[s])
            '''
            if store[s] != new:
                converged = False
            store[s] = new
        return converged


    # TODO might not want to check convergence on both. convergence in one probably 
    # implies convergence in both
    def update_policy(self):
        print('updating policy...')
        # TODO self.pi might not work. may count as partial application.
        return self.compute_check_convergence(self.pi, self.policy)


    def update_values(self):
        print('updating values...')
        return self.compute_check_convergence(self.maxvalue, self.values)

    # TODO might not want to check convergence on both. convergence in one probably 
    # implies convergence in both (doesn't seem to, but i think policy update is broken)

    # TODO it seemed that these two steps are equivalent to single equation 
    # on Wiki under value iteration
    def value_iteration(self, n=None):
        i = 0
        while n is None or i < n:
            pconverged = self.update_policy()
            vconverged = self.update_values()
            print(pconverged, vconverged)
            if pconverged and vconverged:
                print('policy converged after', i, 'iterations')
                break
            i += 1
        return self.policy, self.values

    def check_P(self):
        # TODO is that the right notation?
        """
        Asserts the P(s' | s, a) sums to 1 for each s
        """
        # TODO why do a few rows not sum to 1?
        for s in self.S:
            for a in self.A:
                #print(np.sum(self.P[a,s,:]))
                if np.sum(self.P[a,s,:]) != 1:
                    print(a, s, self.P[a,s,:])
                #assert np.sum(self.P[a,s,:]) == 1

        tot = np.sum(self.P, axis=0)
        print('tot', tot.shape)
        # check out what the diagonal elements are doing
        for s in self.S:
            print(tot[s,s])

        print('sum of all P', np.sum(self.P))
        print('sum of all R', np.sum(self.R))