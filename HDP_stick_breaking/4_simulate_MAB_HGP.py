#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp
from itertools import product
from matplotlib.ticker import MultipleLocator

from misc import *
from environment import MultiArmedBandit
from agent import HibachiGrillProcess
from world import World

max_context = 6

plot_reward_dist = True
plot_context = True
plot_choice = False

dpi = 100
#PP
np.random.seed(1)
# Task setup parameters
na = 3
nb = na
ns = nb+1
no = ns
nr = 3
nc = 1
T = 2
npi = na**(T-1)
switch = 20
reps = 5
# training_protocol = np.tile(np.arange(nb,-1,-1).repeat(switch),reps)
training_protocol = np.tile(np.arange(nb).repeat(switch),reps)


TAU = training_protocol.size

# Agent setup Parameters
h = 1000
approx_pred_pol = True
approx_pred_rew = True

gammas = np.array([5]) #np.arange(1,4,0.1)#      # global prior context opening tendency
rho_global = np.array([1])     # global prior counts forgetting rate

sim_params = product(gammas, rho_global)
reps = 5  # how many times to run simulation with same params

# sim_data = np.zeros([gammas.size*rho_global.size*reps,8])

print(f"-----------------------------------")
print(f"{gammas.size*rho_global.size*reps} simulations to run")
i = -1

for gamma, rho_g in sim_params:
    for rep in range(reps):

        '''           define policies            '''
        # policies = list(product(list(np.arange(na))*(T-1)))
        policies = np.array(list(product( np.arange(na), repeat= T-1)))


        '''       define p(s_t|s_t-1, a_t-1)      '''
        # prior_states = np.array([0,0,1])
        # state_transition_matrix = np.array([[[1,0,1],
        #                                     [0,1,0],
        #                                     [0,0,0]],

        #                                     [[1,0,0],
        #                                     [0,1,1],
        #                                     [0,0,0]]]).transpose(1,2,0)

        prior_states = np.array([0]*nb + [1]) # np.array([0,0,1])
        state_transition_matrix = np.array([np.eye(nb+1)]*nb).transpose([1,2,0])
        state_transition_matrix[:,-1,:] = np.eye(nb+1)[:,:-1]


        '''          define p(o_t|s_t)            '''
        observation_generation_matrix = np.eye(ns)


        '''           define p(r|s,c)             '''
        lambda_H = np.ones([nr,ns])
        lambda_H[-1,-1] = 100
        counts_prior_rewards = np.stack([lambda_H for i in range(nc)],axis=-1)
        
        bias = 10
        
        init_counts = np.ones([nr,nb+1])
        init_counts[0,0] = bias
        init_counts[1,1:] = bias
        init_counts[:,-1] = [1,1,100]
        counts_prior_rewards[:,:,0] = init_counts
        counts_prior_rewards[:,:,0] += np.random.uniform(low=0, high=1, size=(nr,ns))


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

        ''' define counts for stimuli bundling parameter'''
        
        counts_prior_bundle = np.ones(2) # np.array([3,1])# 
        
        '''   define Env reward generation matrix '''
        p = 0.9
        q = 1 - p
        
        
        bandits = np.arange(nb)
        reward_generation_matrix = np.ones([nr,ns,len(bandits)])*q

        for context,b in enumerate(bandits):
            reward_generation_matrix[0,b,context] = p
            reward_generation_matrix[1,np.arange(nb+1) != b,context] = p

        reward_generation_matrix[-1,:] = 0
        reward_generation_matrix[:,-1] = 0
        reward_generation_matrix[-1,-1] = 1



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

        agent = HibachiGrillProcess(lambda_H = lambda_H,
                    TAU=TAU,
                    T=T,
                    gamma=gamma,
                    # alpha=alpha,
                    # kappa=kappa,
                    state_transition_matrix = state_transition_matrix,
                    observation_generation_matrix = observation_generation_matrix,
                    utility = utility,
                    policies = policies,
                    # prior_rewards = prior_rewards,
                    counts_prior_rewards = counts_prior_rewards,
                    prior_policies = prior_policies,
                    counts_prior_policies = counts_prior_policies,
                    counts_prior_bundle = counts_prior_bundle,
                    prior_states = prior_states,
                    na = na,
                    env = env,
                    approx_pred_pol = approx_pred_pol,
                    approx_pred_rew = approx_pred_rew,
                    h = h,
                    debug=False, # If set to True will print inferred agent beliefs up to trial 40?
                    # rho_l= rho_l,
                    rho_g = rho_g,
                    max_context=max_context,
                    K = nc)


        world = World(agent, env)

        world.simulate_experiment()

        save_json(world, 'test.json')
        data = load_json('test.json')
        ##############################

        data = data.agent
        post_policies = np.nan_to_num(data.posterior_policies)
        prior_policies = np.nan_to_num(data.prior_policies)
        like_policies = np.nan_to_num(data.likelihood_policies)
        post_context = data.posterior_context
        actions = data.actions

        post_policies = np.einsum('ktpc,ktc->ktp', post_policies, post_context)
        # prior_policies = np.einsum('kpc,ktc->ktp', prior_policies, post_context)
        # like_policies = np.einsum('ktpc,ktc->ktp', like_policies, post_context)

        plt.style.use('default')

        if agent.K >= 1:  # K is number of inferrred contexts
            

            ### do some labe allocation
            Q_rew = agent.prior_rewards[-1,:,:,:agent.K]  + 1e-10      # inferred reward distribution given state and context
            best_fit = 0
            best_label = []        
            for perm in range(100):    
                l = np.random.permutation(agent.K)        
                labels = l[agent.context]
                fit = (labels == training_protocol).sum()/TAU
                
                if fit > best_fit:
                    best_fit = fit
                    best_label = l
        
            print(f"best label assignment: {best_label}")
           
            plots = [Q_rew[:,:,k] for k in range(agent.K)]
            titles = [None for k in range(agent.K)]
            titles[0] = f"l:{best_label}, gamma:{round(gamma,6)}"

            titles = [f"context {k}" for k in range(agent.K)]
            file_title = f"test"

            ### plots inferred reward distributions for each context
            if plot_reward_dist:
                plot_heatmap(data=plots, file_title=file_title, title=titles,dpi=dpi)
            # plot_heatmap(agent.transition_matrix.round(2),dpi=dpi)


            ### CONTEXT PLOT
            if plot_context:
                data = agent
                post_policies = np.nan_to_num(data.posterior_policies[:,:,:,:data.K])
                prior_policies = np.nan_to_num(data.prior_policies[:-1,:,:data.K])
                like_policies = np.nan_to_num(data.likelihood_policies[:,:,:,:data.K])
                post_context = data.posterior_context[:,:,:data.K+1]
                actions = data.actions

                unexpected_event = np.zeros(TAU)
                
                for trial, trial_type in enumerate(training_protocol):
                    if trial_type == 0:
                        unexpected_event[trial] = agent.observations[trial,1] != agent.rewards[trial,1] 
                    else:
                        unexpected_event[trial] = agent.observations[trial,1] == agent.rewards[trial,1] 

                inf_context = np.argmax(agent.posterior_context[:,1,:],axis=1)
                y_val_unexp_event = agent.posterior_context[np.arange(TAU),1,inf_context]*unexpected_event
                y_val_unexp_event[y_val_unexp_event == 0] = None

                y_val_action = agent.posterior_context[np.arange(TAU),1,inf_context]*agent.actions[:,0]
                y_val_action[y_val_action == 0] = None
                

                K = K = np.cumsum(agent.opened_new_context)+1 
                novel_context = data.posterior_context[np.arange(TAU),:,K[:-1]]
                post_context[np.arange(TAU),:,K[:-1]] = 0
                fig, ax = plt.subplots(1, figsize=(5,3))
                fig.set_dpi(dpi)
                ax.set_ylim((0,1.05))
                plt.grid(axis="x")
                plt.grid(axis="y")
                ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
                # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
                # plt.hlines(0.5,xmin=0,xmax=switch*2, color = 'k', alpha=0.2)
                # ax.scatter(np.arange(TAU), y_val_action, marker="o", color="r", s=30)
                # ax.scatter(np.arange(TAU), y_val_unexp_event, marker="x", color="k")

                df = pd.DataFrame({"trial": np.arange(TAU), "action": agent.actions[:,0], "context":training_protocol})

                for c in range(data.K):
                    ax.plot(post_context[:,1,c],label=f"context {c+1}")
                    ax.legend()
                    
                ax.plot(novel_context[:,1], 'gray', label=f"novel context")
                ax.set_title(fr"$\gamma=${gamma}")

                ax.plot(data.posterior_bundle[:,0],'-.',color="green", alpha=0.5, label="q(w=0)")
                
                new_context = (data.opened_new_context == True).nonzero()

                for ind in new_context:
                    ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5)
                # ax.set_title(f"Posterior Context Iteration: {rep},  correct: {best_fit.round(3)}%")
                ax.set_title(f"Posterior over contexts,  correct: {best_fit.round(3)}%")
                ax.legend(bbox_to_anchor=[1,0.6])#loc="lower right", framealpha=1)

            if plot_choice:
                plt.figure()
                df = pd.DataFrame({"trial": np.arange(TAU), "action": agent.actions[:,0], "context":training_protocol})
                sns.catplot(data=df, x="action",kind="count", hue="context")
