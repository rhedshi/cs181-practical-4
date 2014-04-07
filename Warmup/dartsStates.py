import numpy as np

"Number of states"
num_s = 118

"""
_| _| _| _| _|_
_| 7|12| 1|14|_
_| 2|13| 8|11|_
_|16| 3|10| 5|_
_| 9| 6|15| 4|_
_| _| _| _| _|_

"""
def transModel(s, a):
    """Return the probabilities of reaching the next state given state s
    and action a"""
    probs = np.zeros(num_s)
    new_s = np.array([[0, 0, 0, 0],
                      [1, 8, 12, 14, 0],
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
    probs[s+a]=0.6
    probs[s+new_s[a,1]] += 0.1
    probs[s+new_s[a,2]] += 0.1
    probs[s+new_s[a,3]] += 0.1
    probs[s+new_s[a,4]] += 0.1
    return probs

def reward(s):
    "Return the reward for reaching state s"
    if s < 101:
        return 0
    elif s == 101:
        return 1
    else:
        return -1

