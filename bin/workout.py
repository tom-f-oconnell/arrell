#!/usr/bin/env python3

import gym
import arrell.util
import arrell.mdp
import numpy as np

env = gym.make('FrozenLake-v0')

discounts = np.linspace(0.8, 1, num=40)
avg_values = []
value_stderr = []

for discount in discounts:
    # for debugging. agent is not supposed to have access to these values for comparing algorithms.
    R, P = arrell.util.gym_P_to_RP(env.env.P)
    agent = arrell.mdp.MDP(env.observation_space.n, env.action_space.n, P, R, discount)
    agent.value_iteration()

    episodes = 5000
    max_steps = 1000
    values = []
    for i in range(episodes):
        s0 = env.reset()
        done = False
        V = 0
        for j in range(max_steps):
            # TODO action selection in Q-learning tutorial on medium doesnt make sense.
            # they seem to be acting like actions arent integers. run their code.
            s0, r, done, _ = env.step(agent.policy[s0])
            V += r
            if done:
                values.append(V)
                break
        if not done:
            print('finished without reaching terminal state')

    avg_values.append(np.mean(values))
    value_stderr.append(np.std(values) / np.sqrt(len(values)))

print(avg_values)
print(value_stderr)
