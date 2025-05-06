
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
                 observation_generation_matrix=None, no=None,
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
        self.nb = n_bandits

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

class SpeakerDiscretizationEnvironment():
    """
    Define true underlying HDP-HMM
    We can imagine we are modeling a conversation between three people.
    Each say W=5 words at a time. Every time a speaker finishes speaking,
    a new speaker says another 5 words. The probability of who the next
    speaker depends on who the last speaker was. 

    Each speaker has the same vocabulary of 10 words, which they use with
    different frequencies. So we havea HDP over the distributions over
    words for each speaker? If successfull, the algorithm should infer three speakers 
    with a high probability?
    """
    
    def __init__(self,M=10, K_true=3):
        
        self.M = 10      # number of observation categories
        self.K_true = 3  # true number of mixture components

        k = 8
        weights = np.exp(np.linspace(0,1,10)*k).round(5)
        weights_norm = weights/weights.sum()

        weights  = np.array([.22,.15,.07, 0.04, 0.02])
        weights = np.hstack((np.flip(weights),weights))

        ### define mixture components
        component_params_true = np.zeros([M,K_true])
        component_params_true[:,2] = weights_norm
        component_params_true[:,1] = weights
        component_params_true[:,0] = np.flip(weights_norm)
        self.component_params_true = component_params_true
    
            ### define transition dynamics
        self.transition_matrix_true = np.array([[.7 , .05 , .1],
                                                [.15, .6  , .1],
                                                [.15, .35 , .8]])
        
    def plot_speakers(self):
        plt.plot(self.component_params_true,'-o')
        print(self.component_params_true.sum(axis=0))


    def plot_heatmap(self, data, title=None, vmin=0, vmax=1):
        
            
        fig, ax = plt.subplots(1,1, figsize=(6,6))

        sns.heatmap(data=data, annot=True, annot_kws={"size":16}, cmap="viridis",\
                    cbar=False, fmt='.2f', ax=ax, vmin=vmin, vmax=vmax)
            
        if title is not None:
            ax.set_title(title)

        plt.show()

    def create_data(self,W=50, N=5):

        """Simulate HDP-HMM"""
        self.N = 150
        self.W = 5

        data = np.zeros([N*W,2],dtype=int)   # first column for observed value, second column for what component
        data[0,1] = 1                        # the middle speaker says the first word
        
        for n in range(N):
            for w in range(W):
                i = n*W+w
                speaker = data[i, 1]
                word = np.random.choice(np.arange(self.M), p=self.component_params_true[:,speaker])
                data[i,0] = word
            
                if w < W-1:
                    next_speaker = speaker
                else: 
                    next_speaker = np.random.choice(np.arange(self.K_true), p=self.transition_matrix_true[:, speaker])
                if i != N*W-1:
                    data[i+1,1] = next_speaker

        data = np.hstack((data, np.arange(data.shape[0])[:,None]))
        return data

    def plot_data(self,data):

        plt.figure()
        sns.scatterplot(data=pd.DataFrame(data,columns=["data","component","t"],),
                        x = "t", y="data", hue="component", palette="tab10")
