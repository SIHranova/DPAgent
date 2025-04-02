#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp
from itertools import product

from misc import *
from environment import MultiArmedBandit
from agent import Agent
from world import World
from perception import HierarchicalPerception


np.random.seed(11)
for rep in range(10):

  # Task setup parameters
  na = 5
  nb = na
  ns = nb+1
  no = ns
  nr =  3
  nc = 6
  T = 2
  npi = na**(T-1)
  training_protocol = np.tile(np.arange(5).repeat(100),2)
  # training_protocol = np.concatenate([training_protocol, [3]*100])
  
  # training_protocol = np.repeat(np.arange(2),100)
  TAU = training_protocol.size


  # Agent setup Parameters
  h = 1000
  approx_pred_pol = False
  approx_pred_rew = False


  '''           define policies            '''
  # policies = list(product(list(np.arange(na))*(T-1)))
  policies = np.array(list(product( np.arange(na), repeat= T-1)))


  '''       define p(s_t|s_t-1, a_t-1)      '''
  # prior_states = np.array([0,0,1])
  # state_transition_matrix = np.array([[[1,0,1],
  #                                      [0,1,0],
  #                                      [0,0,0]],

  #                                     [[1,0,0],
  #                                      [0,1,1],
  #                                      [0,0,0]]]).transpose(1,2,0)
  prior_states = np.array([0]*nb + [1]) # np.array([0,0,1])
  state_transition_matrix = np.array([np.eye(nb+1)]*nb).transpose([1,2,0])
  state_transition_matrix[:,-1,:] = np.eye(nb+1)[:,:-1]


  '''          define p(o_t|s_t)            '''
  observation_generation_matrix = np.eye(ns)


  '''           define p(r|s,c)             '''

  # counts = np.array([[1,1,1],
  #                    [1,1,1],
  #                    [1,1,100]])

  # counts = np.array([[10,1,1],
  #                    [1,10,1],
  #                    [1,1,100]])

  # counts_prior_rewards = np.stack( [ counts for i in range(nc) ],axis=-1)

  lambda_H = np.ones([nr,ns])
  lambda_H[-1,-1] = 100
  counts_prior_rewards = np.stack([lambda_H for i in range(nc)],axis=-1)
  counts_prior_rewards += np.random.uniform(low=0,high=0.5, size = counts_prior_rewards.shape)
  bias = 3

  init_counts = np.ones([nr,nb+1])
  init_counts[0,0] = bias
  init_counts[1,1:] = bias
  init_counts[:,-1] = [1,1,100]
  counts_prior_rewards[:,:,0] = init_counts


  if approx_pred_rew:
    prior_rewards = scp.digamma(counts_prior_rewards) - scp.digamma(counts_prior_rewards.sum(axis=0)) 
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
  prior_context = np.array([0.9] + [1-0.9]*(nc-1))

  p = 0.9
  q = 1-p 

  '''   define Env reward generation matrix '''
  bandits = np.arange(nb)
  reward_generation_matrix = np.ones([nr,ns,len(bandits)])*q

  for context,b in enumerate(bandits):
      reward_generation_matrix[0,b,context] = p
      reward_generation_matrix[1,np.arange(nb+1) != b,context] = p

  reward_generation_matrix[-1,:] = 0
  reward_generation_matrix[:,-1] = 0
  reward_generation_matrix[-1,-1] = 1


  # Plot task setup

  '''Plot state transition matrix'''
  if False:
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


  # Run Simulations
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



  # Simulation analysis

  save_json(world, 'test.json')
  data = load_json('test.json')

  data = data.agent.perc
  post_policies = data.posterior_policies
  prior_policies = data.prior_policies
  like_policies = data.likelihood_policies
  post_context = data.posterior_context
  actions = data.actions

  post_policies = np.einsum('ktpc,ktc->ktp', post_policies, post_context)
  prior_policies = np.einsum('ktpc,ktc->ktp', prior_policies, post_context)


  plt.style.use('default')


  plt.figure()
  plt.grid()
  for c in range(nc):
    plt.plot(np.arange(TAU), post_context[:,-1,c], label = f'posterior_context {c}')
  plt.ylim([0,1])
  plt.legend()

  Q_rew = [agent.perc.prior_rewards[-1,0][:,:,k] for k in range(agent.nc)]
  plot_heatmap(Q_rew)


  # plt.figure()
  # plt.grid()

  # for c in range(nc):
  #   for pi in range(na):
  #     print(pi,c)
  #     plt.plot(np.arange(TAU), like_policies[:,0,pi,c],   label = f'like_pol_{pi}_cont_{c}')
  #     # plt.plot(np.arange(TAU), like_policies[:,0,pi,nc],   label = 'like_pol_0_cont_1')
  #     # plt.plot(np.arange(TAU), like_policies[:,0,pi,nc],   label = 'like_pol_1_cont_0')
  #     # plt.plot(np.arange(TAU), like_policies[:,0,pi,nc],   label = 'like_pol_1_cont_1')
  # plt.ylim([0,1])
  # plt.legend(bbox_to_anchor=[1,0.8])

  # like_policies = np.einsum('ktpc,ktc->ktp', like_policies, post_context)
  # plt.figure()
  # plt.grid()
  # # plt.vlines(100,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
  # for pi in range(na):
  #   plt.plot(np.arange(TAU), prior_policies[:,0,pi], label = 'prior $\pi$')
  #   plt.plot(np.arange(TAU), like_policies[:,0,pi] , "-s", markersize=5, label = 'like $\pi$')
  #   plt.plot(np.arange(TAU), post_policies[:,0,pi], "-x", markersize=5, label = 'posterior $\pi$')
  # plt.legend(bbox_to_anchor=[1,0.8])


  # plt.figure()
  # plt.grid()
  # # plt.vlines(100,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
  # plt.plot(np.arange(TAU), actions, '-x',label = 'action')
  # plt.legend()


