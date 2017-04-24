#!/usr/bin/env python3

import gym
import arrell.util
import arrell.mdp
import arrell.qlearn
import numpy as np

def run(env, agent, episodes, max_steps):
    values = []
    for i in range(episodes):
        s0 = env.reset()
        agent.state = s0

        done = False
        V = 0
        for j in range(max_steps):
            # TODO action selection in Q-learning tutorial on medium doesnt make sense.
            # they seem to be acting like actions arent integers. run their code.
            s0, r, done, _ = agent.step(env)
            V += r

            if done:
                values.append(V)
                break
        if not done:
            print('finished without reaching terminal state')

    return values


def stats(values):
    return np.mean(values), np.std(values) / np.sqrt(len(values))


env = gym.make('FrozenLake-v0')

# TODO i tried training on R and P for each reset of frozenlake, and was still getting performance
# below 0.78. is my value iteration incorrect? isn't it gauranteed to converge to optimal policy
# for MDP? isn't frozenlake a MDP? what gives?

# for debugging. agent is not supposed to have access to these values for comparing algorithms.
R, P = arrell.util.gym_P_to_RP(env.env.P)
discount = 0.99
# for updates to P and R
learning_rate = 0.1
value_iter = arrell.mdp.FullMDP(env.observation_space.n, env.action_space.n, \
        P, R, discount, learning_rate)

'''
value_iter.update_P = value_iter.unprincipled_P_update
value_iter.update_R = value_iter.unprincipled_R_update
'''
value_iter.update = lambda a,s,r: None

# build initial policy from initial P and R
value_iter.value_iteration()

discount = 0.99
learning_rate = 0.85
qlearn = arrell.qlearn.QLearner(env.observation_space.n, env.action_space.n, discount, learning_rate)


episodes = 5000
max_steps = 1000
#agents = [value_iter, qlearn]
agents = [qlearn]

for a in agents:
    print(stats(run(env, a, episodes, max_steps)))


