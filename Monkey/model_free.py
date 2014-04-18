# Implements model-free learning (Q-learning)

import numpy as np
import numpy.random as npr
import sys
import math
import random

from SwingyMonkey import SwingyMonkey


class ModelFreeLearner:

    def __init__(self):
        bin_count = 10

        #self.tree_bot_range = (0, 400)
        #self.tree_bot_bins = 10
        #self.tree_top_range = (0, 400)
        #self.tree_top_bins = bin_count
        self.tree_dist_range = (0, 600)
        self.tree_dist_bins = 20
        self.monkey_vel_range = (-50,50)
        self.monkey_vel_bins = 10
        #self.monkey_bot_range = (0, 450)
        #self.monkey_bot_bins = 10
        #self.monkey_top_range = (0, 450)
        #self.monkey_top_bins = bin_count
        self.top_diff_range = (-400, 450)
        self.top_diff_bins = 20

        self.alpha = 0.1
        self.gamma = 0.1
        self.epsilon = 0

        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.iter = 0

        dims = self.basis_dimensions() + (2,)
        self.Q = np.zeros(dims)

        'Number of times taken action a from each state s'
        self.k = np.ones(dims)

    def reset(self):
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.iter += 1

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        #
        if (random.random() < self.epsilon):
            new_action = random.choice((0,1))
        else:
            new_action = np.argmax(self.Q[self.basis(state)])
        new_state  = state

        self.last_action = new_action
        self.last_state  = self.current_state
        self.current_state = new_state

        s  = self.basis(state)
        a  = (self.last_action,)
        self.k[s + a] += 1

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        if (self.last_state != None) and (self.current_state != None) and (self.last_action != None):
            s  = self.basis(self.last_state)
            sp = self.basis(self.current_state)
            a  = (self.last_action,)

            # if self.k[s + a] < 10:
            #     alpha = 0.01
            # else:
            #     alpha = 0.01 / self.k[s + a]

            if self.iter < 100:
                alpha = 0.1
            elif self.iter < 200:
                alpha = 0.01
            elif self.iter < 500:
                alpha = 0.001
            else:
                alpha = 0.0001

            # alpha = self.alpha * (.1 if self.k[s + a] > 100 else 1)
            # alpha = self.alpha * (1 / self.k[s + a]**(3/2))

            self.Q[s + a] = self.Q[s + a] + alpha * (reward + self.gamma * np.max(self.Q[sp]) - self.Q[s + a] )

        self.last_reward = reward


    def bin(self, value, range, bins):
        '''Divides the interval between range[0] and range[1] into equal sized
        bins, then determines in which of the bins value belongs'''
        bin_size = (range[1] - range[0]) / bins
        return math.floor((value - range[0]) / bin_size)

    def basis_dimensions(self):
        '''Returns a tuple containing the dimensions of the state space;
        should match the dimensions of an object returned by self.basis'''
        return (\
            # self.tree_bot_bins, \
             #self.tree_top_bins, \
            self.tree_dist_bins, \
            self.monkey_vel_bins, \
            # self.monkey_bot_bins, \
             #self.monkey_top_bins)#, \
            self.top_diff_bins)

    def basis(self, state):
        '''Accepts a state dict and returns a tuple representing this state;
        used for indexing into self.V, self.R, etc.'''
        return (\
                # self.bin(state["tree"]["bot"],self.tree_bot_range,self.tree_bot_bins),    \
                # self.bin(state["tree"]["top"],self.tree_top_range,self.tree_top_bins),    \
                self.bin(state["tree"]["dist"],self.tree_dist_range,self.tree_dist_bins), \

                self.bin(state["monkey"]["vel"],self.monkey_vel_range,self.monkey_vel_bins), \
                # self.bin(state["monkey"]["bot"],self.monkey_bot_range,self.monkey_bot_bins), \
                #self.bin(state["monkey"]["top"],self.monkey_top_range,self.monkey_top_bins))
                self.bin(state["tree"]["top"]-state["monkey"]["top"],self.top_diff_range,self.top_diff_bins))



def evaluate(gamma=0.4, iters=100, chatter=True):

    learner = ModelFreeLearner()
    learner.gamma = gamma

    highscore = 0
    avgscore = 0.0

    if chatter:
        print "epoch", "\t", "score", "\t", "high", "\t", "avg"

    for ii in xrange(iters):

        learner.epsilon = 1/(ii+1)

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

        if chatter:
            print ii, "\t", score, "\t", highscore, "\t", avgscore

        # Reset the state of the learner.
        learner.reset()

    return -avgscore


def find_hyperparameters():

    # find the best value for hyperparameters
    best_parameters = (0,0)
    best_value = 0
    for gamma in np.arange(0.1,1,0.1):
        parameters = {"gamma": gamma}
        value = evaluate(**parameters)
        if value < best_value:
            best_parameters = parameters
            print "Best: ",parameters, " : ", value


    print best_parameters
    return best_parameters

evaluate(iters=1000,gamma=0.4)
