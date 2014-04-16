# Implements model-free learning (Q-learning)

import numpy as np
import numpy.random as npr
import sys
import math

from SwingyMonkey import SwingyMonkey


class ModelFreeLearner:

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

        self.alpha = 0.5
        self.gamma = 0.5



        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        dims = self.basis_dimensions() + (2,)
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

        new_action = np.argmax(self.Q[self.basis(state)])
        new_state  = state

        self.last_action = new_action
        self.last_state  = self.current_state
        self.current_state = new_state

        s  = self.basis(state)
        a  = (self.last_action,)
        self.k[s + a] += 1

        # print state
        # print self.last_action
        # print self.Q
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        if (self.last_state != None) and (self.current_state != None) and (self.last_action != None):
            s  = self.basis(self.last_state)
            sp = self.basis(self.current_state)
            a  = (self.last_action,)

            # print s + a , " : ", self.Q[s + a]
            # print sp + a, " : ", self.Q[sp + a]
            # print reward
            # print '-------'

            self.Q[s + a] = self.Q[s + a] + self.alpha * (reward + self.gamma * np.max(self.Q[sp]) - self.Q[s + a] )

            # attempt at variable k
            """if self.k[s + a] < 3:
                alpha = 0.5
            else:
                alpha = 1.0 / self.k[s + a]
                """

            self.Q[s + a] = self.Q[s + a] + alpha * (reward + gamma * np.max(self.Q[sp]) - self.Q[s + a] )

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



def evaluate(x, iters=50):

    learner = ModelFreeLearner()
    (learner.alpha, learner.gamma) = x

    highscore = 0
    avgscore = 0.0

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

        score = swing.get_state()['score']
        highscore = max([highscore, score])
        avgscore = (ii*avgscore+score)/(ii+1)
        # print avgscore

        # Reset the state of the learner.
        learner.reset()

    print x, " : ", avgscore
    return -avgscore


# import scipy.optimize
# print scipy.optimize.minimize(evaluate, np.array([0.5, 0.5]), method="COBYLA", bounds=[(0,1),(0,1)], options={ "maxiter":20, "disp":True })


best_parameters = (0,0)
best_value = 0
for alpha in np.arange(0.3,0.4,0.02):
    for gamma in np.arange(0.4,0.5,0.02):
        parameters = (alpha,gamma)
        value = evaluate(parameters)
        if value < best_value:
            best_parameters = parameters
            print "Best: ",parameters, " : ", value


print best_parameters
