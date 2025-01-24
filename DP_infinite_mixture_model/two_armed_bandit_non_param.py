#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp
from itertools import product

from misc import *
from environment import MultiArmedBandit
from agent import NonParamAgent
from world import World
from perception import NonParamHierarchicalPerception


#PP
np.random.seed(324243242)

# Task setup parameters
na = 2
nb = 2
ns = nb+1
no = ns
nr = nb+1
nc = 1
T = 2
npi = na**(T-1)
switch = 10
training_protocol = np.tile(np.arange(2).repeat(20),5)

TAU = training_protocol.size

# Agent setup Parameters
h = 1000
approx_pred_pol = True
approx_pred_rew = True

# kappa = 5

for kappa in np.arange(2.8,3.5,0.1):

  # kappa = 3.2


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

  counts_prior_rewards = np.stack( [ counts for i in range(nc) ],axis=-1)
  # counts_prior_rewards[:,:,0] = np.array([[20,2,1],
  #                                         [2,20,1],
  #                                         [1,1,100]])

  if approx_pred_rew:
    prior_rewards = scp.digamma(counts_prior_rewards) - scp.digamma(counts_prior_rewards.sum(axis=0))
    prior_rewards = scp.softmax(prior_rewards, axis=0)
  else:
    prior_rewards = counts_prior_rewards / counts_prior_rewards.sum(axis=0)


  '''           define counts alpha in p(pi|theta,alpha)           '''
  counts_prior_policies = np.zeros([npi,nc]) + h

  if approx_pred_pol:
    prior_policies = scp.softmax(scp.digamma(counts_prior_policies) - scp.digamma(counts_prior_policies.sum(axis=0)))
  else:
    prior_policies = counts_prior_policies / counts_prior_policies.sum(axis=0)


  '''       define dummy utility RV p(R=1) '''
  utility = np.array([0.99, 0.005,0.005])


  '''       define prior over contexts p(c) '''

  counts_prior_context = np.array([0]*(nc-1) + [kappa])

  p = 1
  prior_context = np.array([p] + [1-p]*(nc-1))             # this is different than the normalized counts over context!

  '''define context transition matrix p(c_t|c_t-1))'''

  if nc == 1:
    context_transition_matrix = np.array([1])
  else:
    p = 0.95
    q = (1-p)/(nc-1)
    context_transition_matrix = np.eye(nc)*(1-2*q) + q

  '''   define Env reward generation matrix '''
  reward_generation_matrix =  np.array([[[0.9,0.1,0],
                                        [0.1,0.9,0],
                                        [0  ,0  ,1]],
                                        [[0.1,0.9,0],
                                        [0.9,0.1,0],
                                        [0  ,0  ,1]]]).transpose(1,2,0)

  #PP Plot task setup
  if False:
    '''Plot state transition matrix'''
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ai, ax,title in zip([0,1], axes, ['$a_1$ = L1', '$a_2$ = L2']):
        ax.xaxis.tick_top()
        sns.heatmap(state_transition_matrix[:,:,ai],annot=True,ax=ax, cbar=False)
        ax.set_xticklabels(['L1','L2','M'],minor=False);
        ax.set_yticklabels(['L1','L2','M'])
        ax.set_title(title);

    '''Plot likelihood matrix'''
    fig, ax = plt.subplots()
    ax = sns.heatmap(prior_rewards[...,0],cbar=False,annot=True)
    ax.set_xticklabels(['L1','L2','M'],minor=False)
    ax.set_yticklabels(['$r_1$', '$r_2$', '$r_3$'])
    ax.set_title('Reward generation beliefs')


    '''Plot Environment reward generation matrix'''
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ai, ax,title in zip([0,1], axes, ['Reward Regime 1', 'Reward Regime 2']):
        ax.xaxis.tick_top()
        sns.heatmap(reward_generation_matrix[:,:,ai],annot=True,ax=ax, cbar=False)
        ax.set_xticklabels(['L1','L2','M'],minor=False)
        ax.set_yticklabels(['$r_1$', '$r_2$', '$r_3$'])
        ax.set_title(title)

  #PP Run Simulations
  env = MultiArmedBandit(state_transition_matrix,
                        reward_generation_matrix, 
                        TAU=TAU,
                        T=T,
                        n_bandits = nb,
                        training_protocol=training_protocol,
                        observation_generation_matrix=observation_generation_matrix,
                        no=no)

  perception = NonParamHierarchicalPerception(
                state_transition_matrix,
                context_transition_matrix,
                observation_generation_matrix,
                utility,
                policies,
                prior_rewards,
                counts_prior_rewards,
                prior_policies,
                counts_prior_policies,
                prior_states,
                counts_prior_context,
                prior_context,
                na,
                nc,
                env,
                approx_pred_pol = approx_pred_pol,
                approx_pred_rew = approx_pred_rew,
                kappa=kappa
              )

  agent = NonParamAgent(
                state_transition_matrix,
                na,
                env,
                perception
              )

  world = World(agent, env)

  world.simulate_experiment()

  save_json(world, 'test.json')
  data = load_json('test.json')
  ##############################

  data = data.agent.perc
  post_policies = np.nan_to_num(data.posterior_policies)
  prior_policies = np.nan_to_num(data.prior_policies)
  like_policies = np.nan_to_num(data.likelihood_policies)
  post_context = data.posterior_context
  actions = data.actions

  post_policies = np.einsum('ktpc,ktc->ktp', post_policies, post_context)
  prior_policies = np.einsum('ktpc,ktc->ktp', prior_policies, post_context)
  like_policies = np.einsum('ktpc,ktc->ktp', like_policies, post_context)

  plt.style.use('default')

  ### context plot
  plt.figure()
  # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
  # plt.hlines(0.5,xmin=0,xmax=switch*2, color = 'k', alpha=0.2)
  for c in range(data.nc):
    #plt.scatter(np.arange(TAU), post_context[:,0,c], label = f'posterior $c$={c}')
    plt.plot(post_context[:,0,c])
  plt.legend()
  plt.title(f"$\kappa=${kappa}")

  new_context = (data.inferred_new_context == True).nonzero()

  for ind in new_context:
    plt.vlines(ind,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)



  # reward entropy plot
  # rewards = data.prior_rewards[:,0,:,:,:]
  # reward_entropy = np.nan_to_num(rewards*np.log(rewards)).sum(axis=1).sum(axis=1)/3

  # plt.figure()
  # plt.grid()

  # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
  # for c in range(data.k):
  #   plt.scatter(np.arange(TAU), reward_entropy[:,c], label = f'entropy $c$={c}')
  # plt.legend()


  # # ACTION PLOT
  # plt.figure()
  # plt.grid()
  # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
  # plt.scatter(np.arange(TAU), actions, label = 'action')
  # plt.legend()



  #### POLICY PLOT
  # plt.figure()
  # plt.grid()
  # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
  # plt.scatter(np.arange(TAU), post_policies[:,0,1], label = 'posterior $\pi$')
  # plt.scatter(np.arange(TAU), prior_policies[:,0,1], label = 'prior $\pi$')
  # plt.scatter(np.arange(TAU), like_policies[:,0,1], label = 'like $\pi$')
  # plt.legend()





  print(data.nc)
  for i in range(data.nc):
      print(data.prior_rewards[-1, 0,:,:,i].round(3),'\n')





# PP

# %%
