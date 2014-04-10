import numpy as np
import dartsStates as darts

#Number of states
num_s = 118
s_stop = 101
num_a = 16

def policyIter(gamma=0.5):
    pi = initPolicy()
    while True:
        piOld = pi

        #Solve system for V(s)
        v = solveV(pi, gamma)

        q = np.zeros((num_s,num_a))
        for s in range(s_stop):
            for a in range(num_a):
                q[s,a] = gamma * np.dot(darts.transModel(s,a), v)

        #Take the index of the maximum Q(s,a)
        pi = np.argmax(q, 1)

        #convergence condition
        if np.array_equal(piOld,pi):
            break
    return pi

def initPolicy():
    "Initalize pi, the policy for a state s"
    return np.zeros(num_s)

"""
Input:
    pi:
"""
def solveV(pi, gamma=0.5):
    "Calculate the value of each state given the policies"
    v = np.zeros(num_s)

    #These are set because they end the game
    v[101] = 1
    v[102:num_s-1] = -1

    #These are subarrays of s and pi up to s_stop
    sSub = np.arange(s_stop, dtype=float)
    piSub = pi[:s_stop]

    trans = darts.transModel(sSub, piSub)
    a = trans - np.identity(num_s)[:s_stop]/gamma

    b = trans[:,101] - np.sum(trans[:,102:117],1)

    v = np.linalg.lstsq(a,b)[0]

    return v