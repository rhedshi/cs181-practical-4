# Implements model-free learning (Q-learning)

import numpy as np
import numpy.random as npr
import sys
import math

from SwingyMonkey import SwingyMonkey


alpha = 0.1
gamma = 0.1

tree_bot_range = (0, 400)
tree_bot_bins = 10
tree_top_range = (0, 400)
tree_top_bins = 10
tree_dist_range = (0, 600)
tree_dist_bins = 10
monkey_vel_range = (-50,50)
monkey_vel_bins = 10
monkey_bot_range = (0, 450)
monkey_bot_bins = 10
monkey_top_range = (0, 450)
monkey_top_bins = 1


def bin(value, range, bins):
    bin_size = (range[1] - range[0]) / bins
    return math.floor((value - range[0]) / bin_size)

def basis_dimensions():
    return (\
        tree_bot_bins, tree_top_bins, tree_dist_bins, \
        monkey_vel_bins, monkey_bot_bins, monkey_top_bins)

def basis(state):
    return (\
            bin(state["tree"]["bot"],tree_bot_range,tree_bot_bins),    \
            bin(state["tree"]["top"],tree_top_range,tree_top_bins),    \
            bin(state["tree"]["dist"],tree_dist_range,tree_dist_bins), \

            bin(state["monkey"]["vel"],monkey_vel_range,monkey_vel_bins), \
            bin(state["monkey"]["bot"],monkey_bot_range,monkey_bot_bins), \
            bin(state["monkey"]["top"],monkey_top_range,monkey_top_bins))


class Learner:

    def __init__(self):
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        dims = basis_dimensions() + (2,)
        self.Q = np.zeros(dims)

        'Number of times taken action a from each state s'
        self.k = np.zeros(dims)

    def reset(self):
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        new_action = np.argmax(self.Q[basis(state)])
        new_state  = state

        self.last_action = new_action
        self.last_state  = self.current_state
        self.current_state = new_state

        """"s  = basis(state)
        a  = (self.last_action,)
        self.k[s + a] += 1"""

        # print state
        # print self.last_action
        # print self.Q
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        if (self.last_state != None) and (self.current_state != None) and (self.last_action != None):
            s  = basis(self.last_state)
            sp = basis(self.current_state)
            a  = (self.last_action,)

            # print s + a , " : ", self.Q[s + a]
            # print sp + a, " : ", self.Q[sp + a]
            # print reward
            # print '-------'

            # attempt at variable k
            """if self.k[s + a] < 3:
                alpha = 0.5
            else:
                alpha = 1.0 / self.k[s + a]"""

            self.Q[s + a] = self.Q[s + a] + alpha * (reward + gamma * np.max(self.Q[sp]) - self.Q[s + a] )

        self.last_reward = reward


iters = 1000
learner = Learner()
highscore = 0

for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    highscore = max([highscore, swing.get_state()['score']])
    print highscore
    # Reset the state of the learner.
    learner.reset()




