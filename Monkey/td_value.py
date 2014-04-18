# Implements model-free learning (Q-learning)

import numpy as np
import numpy.random as npr
import sys
import math
import random

from SwingyMonkey import SwingyMonkey


class TDValueLearner:

    def __init__(self):

        # ranges of each possible dimension for the state space
        bin_count = 10

        # self.tree_bot_range = (0, 400)
        # self.tree_bot_bins = 10
        self.tree_top_range = (0, 400)
        self.tree_top_bins = bin_count
        self.tree_dist_range = (0, 600)
        self.tree_dist_bins = bin_count
        self.monkey_vel_range = (-50,50)
        self.monkey_vel_bins = bin_count
        # self.monkey_bot_range = (0, 450)
        # self.monkey_bot_bins = 10
        self.monkey_top_range = (0, 450)
        self.monkey_top_bins = bin_count
        self.top_diff_range = (-400, 450)
        self.top_diff_bins = 10

        # default values for hyperparameters
        self.alpha = 0.1
        self.gamma = 0.1
        self.epsilon = 0.1

        # state of MDP
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # dimensions of s
        dims = self.basis_dimensions()

        # learned value of state s
        self.V = np.zeros(dims)

        # learned reward of state s
        self.R = np.zeros(dims + (2,))

        # empirical distribution for estimating transition model
        # self.N[s + a] = number of times we've taken action a from state s
        self.N = np.ones(dims + (2,))

        # self.Np[s + a + sp] = number of times we've transitioned to state sp
        # after taking action a in state s
        self.Np = np.zeros(dims + (2,) + dims)

        # note that to calculate the empirical distribution of the transition model P(sp | s,a),
        # you can do:
        #     self.Np[ s + a + (Ellipsis,) ] / self.N[(Ellipsis,) + a]

        'Number of times taken action a from each state s'
        self.k = np.ones(dims + (2,))

    def reset(self):
        # reset state of MDP
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''


        # store state, last state for learning in reward_callback
        self.last_state  = self.current_state
        self.current_state = state
        s  = self.basis(state)

        # plan
        if (random.random() < self.epsilon):
            # with some probability self.epsilon, just pick a random action
            new_action = random.choice((0,1))
        else:
            # otherwise plan based on the learned transition model
            # array of expected values for each possible action
            expected_values = np.array([ np.dot( (self.Np[ s + a + (Ellipsis,) ] / self.N[(Ellipsis,) + a]).flat, self.V.flat ) for a in [(0,), (1,)] ])

            # pick the new action pi(s) as the action with the largest expected value
            new_action =  np.argmax(self.R[s + (Ellipsis,)] + expected_values)

        # store last action, record exploration
        self.last_action = new_action
        a  = (self.last_action,)
        self.k[s + a] += 1

        # learn the transition model
        if (self.last_state != None):
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
            sp = self.basis(self.current_state)
            a  = (self.last_action,)

            # lower alpha over time as we visit more frequently
            # alpha = 1.0 / self.k[s + a]
            alpha = 0.1

            # update V
            self.V[s] = self.V[s] + alpha * ( (reward + self.gamma * self.V[sp]) - self.V[s] )

            # update R with a "running average"
            self.R[s + a] = (self.R[s + a] * (self.k[s + a] - 1) + reward) / (self.R[s + a] + 1)


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
            #self.tree_top_bins,
            self.tree_dist_bins, \
            self.monkey_vel_bins, \
            # self.monkey_bot_bins, \
            # self.monkey_top_bins, \
            self.top_diff_bins)

    def basis(self, state):
        '''Accepts a state dict and returns a tuple representing this state; 
        used for indexing into self.V, self.R, etc.'''
        return (\
                # self.bin(state["tree"]["bot"],self.tree_bot_range,self.tree_bot_bins),    \
                #self.bin(state["tree"]["top"],self.tree_top_range,self.tree_top_bins),    \
                self.bin(state["tree"]["dist"],self.tree_dist_range,self.tree_dist_bins), \

                self.bin(state["monkey"]["vel"],self.monkey_vel_range,self.monkey_vel_bins), \
                # self.bin(state["monkey"]["bot"],self.monkey_bot_range,self.monkey_bot_bins), \
                #self.bin(state["monkey"]["top"],self.monkey_top_range,self.monkey_top_bins),
                self.bin(state["tree"]["top"]-state["monkey"]["top"],self.top_diff_range,self.top_diff_bins))



def evaluate(gamma=0.4, iters=100, chatter=True):

    learner = TDValueLearner()
    learner.gamma = gamma

    highscore = 0
    avgscore = 0.0

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
            print ii, score, highscore, avgscore

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
