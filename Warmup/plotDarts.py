import numpy as np
import matplotlib.pyplot as plt

s_stop = 101

def bar(y):
    x = np.arange(s_stop)

    plt.bar(x, y[:s_stop])

    plt.xlim(0,s_stop)
    plt.ylim(ymin = 0)
    plt.xlabel('State (score)')
    plt.ylabel('Value')

    plt.show()