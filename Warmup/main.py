import numpy as np
import policyIter as policy
import plotDarts as plot

gamma = 1

pi = policy.policyIter(gamma)
v = policy.solveV(pi, gamma)

print pi, v

plot.bar(v)

#85 has the highest value