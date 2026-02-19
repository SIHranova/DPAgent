
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from misc import plot_heatmap

class MultiArmedBandit():

    def __init__(self,
                 state_transition_matrix,
                 reward_generation_matrix,
                 training_protocol=None,
                 TAU=3,
                 T=2,
                 n_bandits=2,
                 observation_generation_matrix=None, 
                 no=None,
                 context_observation_generation_matrix = None):

        self.Rho = reward_generation_matrix
        self.B = state_transition_matrix
        self.nr, self.ns, self.number_reward_regimes = reward_generation_matrix.shape
        self.na =  self.B.shape[-1]
        self.states  = np.zeros([TAU,T],dtype=int)
        self.rewards = np.zeros([TAU,T],dtype=int)
        self.actions = np.zeros([TAU, T-1],dtype=int)
        self.observations = np.zeros([TAU,T],dtype=int)
        self.context_observation_generation_matrix = context_observation_generation_matrix
        self.nco = context_observation_generation_matrix.shape[0]
        self.context_observations = np.zeros([TAU,T],dtype=int)

        self.TAU = TAU
        self.T = T
        self.initial_state = n_bandits

        if observation_generation_matrix is None:
            self.observation_generation_matrix = np.eye(self.ns)
            self.no = self.ns.copy()
        else:
            self.observation_generation_matrix = observation_generation_matrix
            self.no = no


        if training_protocol is None:
            self.training_protocol = np.repeat(np.arange(self.number_reward_regimes),\
                                               np.ceil(self.TAU/self.number_reward_regimes))[:self.TAU]
        else:
            self.training_protocol = training_protocol

    
    def sample_reward(self, t, tau, state):
        regime = self.training_protocol[tau]
        self.rewards[tau,t] = np.random.choice(np.arange(self.nr), p=self.Rho[:,state,regime])
        return self.rewards[tau,t]
    

    def sample_hidden_state(self, t, tau, action):
        self.actions[tau,t-1] = action
        self.states[tau,t] = np.random.choice(np.arange(self.ns), p=self.B[:,self.states[tau, t-1], action])
        
        return self.states[tau,t]
    
    
    def generate_observation(self, t, tau, state):

        self.observations[tau,t] = np.random.choice(np.arange(self.no), p=self.observation_generation_matrix[:,state])
        return self.observations[tau,t]
    
    def generate_context_observation(self, t, tau, context):

        self.context_observations[tau,t] = np.random.choice(np.arange(self.nco), p=self.context_observation_generation_matrix[:,context])
        return self.context_observations[tau,t]

    def initialize_hidden_state(self, tau, starting_state=2):
        self.states[tau,0] = starting_state
        return starting_state
