#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from itertools import product

from environment import MultiArmedBandit
from agent import Agent
from world import World

#%%

T = 5
Lx = 4
Ly = 5
ns = Ly*Lx
no = Ly*Lx
nr = 2
nc = ns
actions = np.array([[0,-1], [1,0], [0,1]])#[-1,0],
na = actions.shape[0]
g1 = 14
g2 = 10
start = 2
TAU = 200



'''           define policies            '''
# policies = list(product(list(np.arange(na))*(T-1)))
policies = list(product( np.arange(na), repeat= T-1))

'''       define p(s_t|s_t-1, a_t-1)      '''
prior = np.array([0,0,1])
state_transition_matrix = np.array([[[1,0,1],
                                     [0,1,0],
                                     [0,0,0]],
                                    [[1,0,0],
                                     [0,1,1],
                                     [0,0,0]]]).transpose(1,2,0)


'''             define p(r|s,c)             '''
counts = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,100]])

counts_likelihood_matrix = np.stack( [ counts for i in range(nc) ],axis=-1)
likelihood_matrix = counts_likelihood_matrix / counts_likelihood_matrix.sum(axis=0)

'''       define dummy utility RV p(R=1) '''
utility = np.array([0.8, 0.15,0.05])


'''   define Env reward generation matrix '''
reward_generation_matrix =  np.array([[[0.9,0.1,0],
                                      [0.1,0.9,0],
                                      [0  ,0  ,1]],
                                     [[0.1,0.9,0],
                                      [0.9,0.1,0],
                                      [0  ,0  ,1]]]).transpose(1,2,0)

#%%

agent = Agent()
env = MultiArmedBandit(state_transition_matrix, reward_generation_matrix, TAU=TAU, T=T, n_bandits = nb)
world = World(agent, env)


world.simulate_experiment()
#%%

'''Plot state transition matrix'''

fig,axes = plt.subplots(1,2,figsize=(10,4))

for ai, ax,title in zip([0,1], axes, ['$a_1$ = L1', '$a_2$ = L2']):
    ax.xaxis.tick_top()
    sns.heatmap(state_transition_matrix[:,:,ai],annot=True,ax=ax, cbar=False);
    ax.set_xticklabels(['L1','L2','M'],minor=False);
    ax.set_yticklabels(['L1','L2','M']);
    ax.set_title(title);

'''Plot likelihood matrix'''

fig, ax = plt.subplots()
ax = sns.heatmap(likelihood_matrix[...,0],cbar=False,annot=True);
ax.set_xticklabels(['L1','L2','M'],minor=False);
ax.set_yticklabels(['$r_1$', '$r_2$', '$r_3$'])
ax.set_title('Reward generation beliefs')


'''Plot Environment reward generation matrix'''

fig,axes = plt.subplots(1,2,figsize=(10,4))
for ai, ax,title in zip([0,1], axes, ['Reward Regime 1', 'Reward Regime 2']):
    ax.xaxis.tick_top()
    sns.heatmap(reward_generation_matrix[:,:,ai],annot=True,ax=ax, cbar=False);
    ax.set_xticklabels(['L1','L2','M'],minor=False);
    ax.set_yticklabels(['$r_1$', '$r_2$', '$r_3$']);
    ax.set_title(title);


