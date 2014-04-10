import numpy as np


#Number of states
num_s = 118

"""
_| _| _| _| _|_
_| 7|12| 1|14|_
_| 2|13| 8|11|_
_|16| 3|10| 5|_
_| 9| 6|15| 4|_
_| _| _| _| _|_

Input:
    s:    state, 0 to (num_s-1)
    a:    action, 0 to 15
Return:
    probs:    array of size num_s with probs[i] indicating probability of
                state i
"""
def transModel1(s, a):
    """Return the probabilities of reaching the next state given state s
    and action a"""

    probs = np.zeros(num_s)

    """new_s[i] are possible states for action i, where new_s[i,0] has 60%
    probability and new_s[i,j] for j=1,2,3 has 10% probability each"""
    new_s = np.array([[1, 8, 12, 14, 0],
                      [2, 7, 13, 16, 0],
                      [3, 6, 10, 16, 0],
                      [4, 5, 10, 15, 0],
                      [5, 4, 10, 11, 0],
                      [6, 3, 9, 15, 0],
                      [7, 2, 12, 0, 0],
                      [8, 1, 10, 11, 13],
                      [9, 6, 16, 0, 0],
                      [10, 3, 5, 8, 15],
                      [11, 5, 8, 14, 0],
                      [12, 1, 7, 13, 0],
                      [13, 2, 3, 8, 12],
                      [14, 1, 11, 0, 0],
                      [15, 4, 6, 10, 0],
                      [16, 2, 3, 9, 0]])
    probs[s+new_s[a,0]] = 0.6
    probs[s+new_s[a,1]] += 0.1
    probs[s+new_s[a,2]] += 0.1
    probs[s+new_s[a,3]] += 0.1
    probs[s+new_s[a,4]] += 0.1
    return probs

"""
Input:
    s: array of states, each element 0 to (num_s-1)
    a: array of actions, each element is 0 to 15
    s and a should be same size
Return:
    probs: array size (a.size, num_s) with probs[i,j] = probability
            of reaching state j with action i
"""
def transModel(s, a):
    "Same as transmodel, but a is an array of actions"

    if type(a) is int:
        return transModel1(s, a)

    if a.size != s.size:
        raise Exception('s and a should be same size')

    #Naive implementation. TODO: vectorize
    probs = np.zeros(((a.size), num_s))
    for i, (ai, si) in enumerate(zip(a,s)):
        probs[i] = transModel1(si,ai)

    return probs

def reward(s):
    "Return the reward for reaching state s"
    if s < 101:
        return 0
    elif s == 101:
        return 1
    else:
        return -1

