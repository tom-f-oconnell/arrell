
import numpy as np


# TODO check P represents valid probabilities leaving each state (sum to 1)
def gym_P_to_RP(gym_P):
    """
    Takes the old OpenAI gym P variable associated with an environment, in format:
        dict[state] -> dict[action] -> list of (p, new state, reward, terminal?)
    to matrices R: (A x S x S) -> real,  P: (A x S x S) -> [0,1]
    """
    smax = 0
    amax = 0
    # TODO are all transition probabilities off of terminal states zero?
    seen_s0 = set()
    for s0, actions in gym_P.items():
        assert s0 not in seen_s0, 'duplicate s0 labels'
        seen_s0.add(s0)

        seen_a = set()
        for a, new_states in actions.items():
            assert a not in seen_a, 'duplicate action labels'
            seen_a.add(a)

            for p, s1, reward, term in new_states:
                if a > amax:
                    amax = a
                if s0 > smax:
                    smax = s0
    Na = amax + 1
    Ns = smax + 1
    R = np.zeros((Na, Ns, Ns))
    P = np.zeros((Na, Ns, Ns))

    for s0, actions in gym_P.items():
        for a, new_states in actions.items():
            total = 0

            expected_rewards = dict()
            normalization = dict()
            for p, s1, reward, term in new_states:
                # TODO how to handle cases where you only get a reward some portion of the time?
                # for a given (s0, a, s1)? reason i cant replace reward with expected value?
                if s1 not in expected_rewards:
                    expected_rewards[s1] = reward * p
                    normalization[s1] = p
                else:
                    expected_rewards[s1] += reward * p
                    normalization[s1] += p

                # only (maybe) works if we replace R entry by expected reward of (s0, a, s1)
                P[a, s0, s1] += p
                total += p
            assert total == 1

            # TODO test
            for s1 in expected_rewards.keys():
                R[a, s0, s1] = expected_rewards[s1] / normalization[s1]

    for s in range(Ns):
        for a in range(Na):
            tot = np.sum(P[a, s, :])
            if tot != 1:
                print(a, s, P[a, s, :])

    return R, P


def show_frozenlake(d):
    rows = 4
    cols = 4
    for i in range(rows):
        for j in range(cols):
            state = i * cols + j
            print(d[state], end=' ')
        print('')
    print('')


