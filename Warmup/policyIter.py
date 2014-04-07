import numpy as np
import dartsStates as darts

"Number of states"
num_s = 118
num_a = 17

def policyIter(gamma=0.5):
    pi = initPolicy()
    while True:
        piOld = pi

        "Solve system for V(s)"

        q = np.zeros((s,a))
        for s in range(num_s):
            for a in range(num_a):
                q[s,a] = gamma * np.dot(darts.transModel(s,a), v)

        pi = np.argmax(q, 1)

        if piOld == pi:
            break
    return pi

def initPolicy():
    return np.ones(num_s)