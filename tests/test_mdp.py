
import numpy as np
import arrell

na = 2
ns = 4
P = np.zeros((na, ns, ns))
R = np.zeros((na, ns, ns))
discount = 1

P[

a = arrell.MDP(ns, na, P, R, discount)
a.policy = {0:
