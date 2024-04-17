#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp

from itertools import product

from environment import MultiArmedBandit
from agent import Agent
from world import World
from perception import HierarchicalPerception

#%%
np.random.seed(11)
# Task setup parameters
na = 2
nb = 2
ns = nb+1
no = ns
nr = nb+1
nc = 2
T = 2
npi = na**(T-1)
training_protocol = np.arange(2).repeat(100)
TAU = training_protocol.size


# Agent setup Parameters
h = 1
approx_pred_pol = False
approx_pred_rew = True


'''           define policies            '''
# policies = list(product(list(np.arange(na))*(T-1)))
policies = np.array(list(product( np.arange(na), repeat= T-1)))


'''       define p(s_t|s_t-1, a_t-1)      '''
prior_states = np.array([0,0,1])
state_transition_matrix = np.array([[[1,0,1],
                                     [0,1,0],
                                     [0,0,0]],

                                    [[1,0,0],
                                     [0,1,1],
                                     [0,0,0]]]).transpose(1,2,0)


'''          define p(o_t|s_t)            '''
observation_generation_matrix = np.eye(ns)


'''           define p(r|s,c)             '''

counts = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,100]])

# counts = np.array([[10,1,1],
#                    [1,10,1],
#                    [1,1,100]])

counts_prior_rewards = np.stack( [ counts for i in range(nc) ],axis=-1)

if approx_pred_rew:
  prior_rewards = scp.digamma(counts_prior_rewards) - scp.digamma(counts_prior_rewards.sum(axis=0))   # checked mentally if broadcasting correct  
  prior_rewards = scp.softmax(prior_rewards, axis=0)
else:
  prior_rewards = counts_prior_rewards / counts_prior_rewards.sum(axis=0)


'''              define p(pi|c)           '''
counts_prior_policies = np.zeros([npi,nc]) + h

if approx_pred_pol:
  prior_policies = scp.softmax(scp.digamma(counts_prior_policies) - scp.digamma(counts_prior_policies.sum(axis=0)))
else:
  prior_policies = counts_prior_policies / counts_prior_policies.sum(axis=0)


'''       define dummy utility RV p(R=1) '''
utility = np.array([0.99, 0.005,0.005])


'''       define prior over contexts p(c) '''
p = 0.9
prior_context = np.array([p] + [1-p]*(nc-1))


'''   define Env reward generation matrix '''
reward_generation_matrix =  np.array([[[0.9,0.1,0],
                                      [0.1,0.9,0],
                                      [0  ,0  ,1]],
                                     [[0.1,0.9,0],
                                      [0.9,0.1,0],
                                      [0  ,0  ,1]]]).transpose(1,2,0)



#%%
env = MultiArmedBandit(state_transition_matrix,
                       reward_generation_matrix, 
                       TAU=TAU,
                       T=T,
                       n_bandits = nb,
                       training_protocol=training_protocol,
                       observation_generation_matrix=observation_generation_matrix,
                       no=no)

perception = HierarchicalPerception(
              state_transition_matrix,
              utility,
              policies,
              prior_rewards,
              counts_prior_rewards,
              prior_policies,
              counts_prior_policies,
              prior_states,
              prior_context,
              na,
              nc,
              env,
              approx_pred_pol = approx_pred_pol,
              approx_pred_rew = approx_pred_rew,
              observation_generation_matrix = observation_generation_matrix
            )

agent = Agent(
              state_transition_matrix,
              na,
              nc,
              env,
              perception
            )

world = World(agent, env)


world.simulate_experiment()
for tau in range(TAU): 
  print(tau,world.agent.rewards[tau,-1],world.agent.actions[tau],'\n', world.agent.perc.posterior_contexts[tau])

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
ax = sns.heatmap(prior_rewards[...,0],cbar=False,annot=True);
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

