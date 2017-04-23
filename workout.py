#!/usr/bin/env python3

import gym
import arrell.util
import arrell.mdp

env = gym.make('FrozenLake-v0')
s0 = env.reset()

#discount = 0.99
discount = 1
# for debugging. agent is not supposed to have access to these values for comparing algorithms.
R, P = arrell.util.gym_P_to_RP(env.env.P)

agent = arrell.mdp.MDP(env.observation_space.n, env.action_space.n, P, R, discount)
print('starting policy', agent.policy)

agent.value_iteration()
print('policy upon convergence', agent.policy)

'''
for s in agent.S:
    for a in agent.A:
        for s1 in agent.S:
            print(agent.R[a, s, s1])
'''

'''
V = 0
for _ in range(1000):
    env.render()
    # TODO action selection in Q-learning tutorial on medium doesnt make sense.
    # they seem to be acting like actions arent integers. run their code.
    s0, r, done, _ = env.step(agent.policy[s0])
    V += r
    if done:
        print('done! V=' + str(V))
'''
