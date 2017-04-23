
import numpy as np
import arrell.mdp

def test_simple_policy_change():
    na = 2
    ns = 3
    P = np.zeros((na, ns, ns))
    R = np.zeros((na, ns, ns))
    discount = 1

    # only two valid actions
    P[0, 0, 1] = 1
    P[1, 0, 2] = 1

    # only one with a reward
    R[0,0,1] = 1
    R[0,1,2] = 0

    # initial policy with action towards no-reward state
    a = arrell.mdp.MDP(ns, na, P, R, discount)
    a.policy = {0: 1, 1: -1, 2: -1}
    assert a.policy[0] == 1

    """
    if algorithm is 
    loop until 'convergence': update policy -> update values
    (with initial values (V) all 0)

    these should be calculations:
    pi(s) = argmax_a{sum over s'{P(a,s,s')(R(a,s,s') + discount * V(s')}}
    V(s) = sum over s'{P(pi(s),s,s')(R(pi(s),s,s') + discount * V(s')}

    i=1
    pi(0) = argmax_a{P(0,0,0)R(0,0,0) + P(0,0,1)R(0,0,1) + P(0,0,2)R(0,0,2), ...}
          = argmax_a{0 + 1 + 0, 0}
          = 0
    (can't take actions from either s=1 or s=2)
    V(0) = 1

    should converge on next iteration because s=0 has no probability of going back to itself
    """
    iterations, policy, values = a.value_iteration()
    assert iterations == 2
    assert a.policy[0] == 0
    assert a.values[0] == 1
    assert a.values[1] == 0
    assert a.values[2] == 0
