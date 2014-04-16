import numpy as np
import numpy.random as npr
import sys
import math

from SwingyMonkey import SwingyMonkey

class ModelBasedLearner:

    def __init__(self):
        self.tree_bot_range = (0, 400)
        self.tree_bot_bins = 10
        self.tree_top_range = (0, 400)
        self.tree_top_bins = 10
        self.tree_dist_range = (0, 600)
        self.tree_dist_bins = 10
        self.monkey_vel_range = (-50,50)
        self.monkey_vel_bins = 10
        self.monkey_bot_range = (0, 450)
        self.monkey_bot_bins = 10
        self.monkey_top_range = (0, 450)
        self.monkey_top_bins = 10


        dims = self.basis_dimensions()
        self.N = np.zeros(dims + (2,))
        self.R = np.zeros(dims + (2,))
        self.Np = np.zeros(dims + (2,) + dims)

        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

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

        new_action = 0

        if self.N[basis(state)].any() == 0:
            new_action = np.argmax(self.R[self.basis(state)])
        else:
            new_action = np.argmax(self.R[self.basis(state)]/self.N[basis(state)])

        new_state  = state

        self.last_action = new_action
        self.last_state  = self.current_state
        self.current_state = new_state

        s  = self.basis(self.last_state)
        sp = self.basis(self.current_state)
        a  = (self.last_action,)

        self.N[s + a] += 1
        self.Np[s + a + sp] += 1

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        if (self.last_state != None) and (self.current_state != None) and (self.last_action != None):
            s  = self.basis(self.last_state)
            a  = (self.last_action,)

            self.R[s + a] += self.last_reward
        
        self.last_reward = reward

    def bin(self, value, range, bins):
        bin_size = (range[1] - range[0]) / bins
        return math.floor((value - range[0]) / bin_size)

    def basis_dimensions(self):
        return (\
            self.tree_bot_bins, self.tree_top_bins, self.tree_dist_bins, \
            self.monkey_vel_bins, self.monkey_bot_bins, self.monkey_top_bins)

    def basis(self, state):
        return (self.bin(state["tree"]["bot"],self.tree_bot_range,self.tree_bot_bins),    \
                self.bin(state["tree"]["top"],self.tree_top_range,self.tree_top_bins),    \
                self.bin(state["tree"]["dist"],self.tree_dist_range,self.tree_dist_bins), \

                self.bin(state["monkey"]["vel"],self.monkey_vel_range,self.monkey_vel_bins), \
                self.bin(state["monkey"]["bot"],self.monkey_bot_range,self.monkey_bot_bins), \
                self.bin(state["monkey"]["top"],self.monkey_top_range,self.monkey_top_bins))

iters = 100
learner = ModelBasedLearner()

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

    # Reset the state of the learner.
    learner.reset()