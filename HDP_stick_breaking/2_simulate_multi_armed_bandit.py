#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp
from itertools import product
from matplotlib.ticker import MultipleLocator
from itertools import product

from misc import *
from environment import MultiArmedBandit
from agent import HDP
from world import World

plt.rcParams['figure.dpi'] = 100
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
switch = 20
training_protocol = np.tile(np.arange(2).repeat(switch),5)
# plt.rcParams['axes.xaxis.major.locator'] = MultipleLocator(switch)

TAU = training_protocol.size

# Agent setup Parameters
h = 1000
approx_pred_pol = True
approx_pred_rew = True


gammas = np.arange(1,2,0.05)
alphas = np.arange(1,2.7,0.05) #[1.9]#[2.1]
kappas = np.arange(0.1,1.5,0.1)

# gammas = np.array([1.7])
# alphas = np.array([1.8])
# kappas = np.array([0.5])
sim_params = product(alphas, gammas, kappas)
reps = 5


sim_data = np.zeros([np.array(alphas.size)*kappas.size*gammas.size*reps,6])

debug = [True, False, False, True,False]
print(f"-----------------------------------")
print(f"{alphas.size*kappas.size*gammas.size*reps} simulations to run")
i = -1
for alpha, gamma, kappa in sim_params:
    for rep in range(reps):

        print("#######################################################")

        i+=1

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

        counts_prior_rewards = np.stack([counts for i in range(nc)],axis=-1)
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


        # '''       define prior over contexts p(c) '''
        # counts_prior_context = np.array([0]*(nc-1) + [kappa])

        # p = 1
        # prior_context = np.array([p] + [1-p]*(nc-1))             # this is different than the normalized counts over context!

        # '''define context transition matrix p(c_t|c_t-1))'''

        # if nc == 1:
        #     context_transition_matrix = np.array([1])
        # else:
        #     p = 0.95
        #     q = (1-p)/(nc-1)
        #     context_transition_matrix = np.eye(nc)*(1-2*q) + q

        '''   define Env reward generation matrix '''
        reward_generation_matrix =  np.array([[[0.9, 0.1, 0], 
                                               [0.1, 0.9, 0], 
                                               [0  , 0  , 1]], 
                                            
                                              [[0.1, 0.9, 0], 
                                               [0.9, 0.1, 0], 
                                               [0  , 0  , 1]]]).transpose(1,2,0)


        #PP Plot task setup
        if False:
            '''Plot state transition matrix'''
            fig,axes = plt.subplots(1,2,figsize=(6,3))
            for ai, ax,title in zip([0,1], axes, ['$a_1$ = L1', '$a_2$ = L2']):
                ax.xaxis.tick_top()
                sns.heatmap(state_transition_matrix[:,:,ai],annot=True,ax=ax, cbar=False, cmap="viridis")
                ax.set_xticklabels(['L1','L2','M'],minor=False);
                ax.set_yticklabels(['L1','L2','M'])
                ax.set_title(title);

            '''Plot likelihood matrix'''
            fig, ax = plt.subplots(figsize=(3,3))
            ax = sns.heatmap(prior_rewards[...,0],cbar=False,annot=True, cmap="viridis")
            ax.set_xticklabels(['L1','L2','M'],minor=False)
            ax.set_yticklabels(['$r_1$', '$r_2$', '$r_3$'])
            ax.set_title('Reward generation beliefs')


            '''Plot Environment reward generation matrix'''
            fig,axes = plt.subplots(1,2,figsize=(6,3))
            for ai, ax,title in zip([0,1], axes, ['Reward Regime 1', 'Reward Regime 2']):
                ax.xaxis.tick_top()
                sns.heatmap(reward_generation_matrix[:,:,ai],annot=True,ax=ax, cbar=False, cmap="viridis")
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


        agent = HDP(lambda_H = counts,
                    TAU=TAU,
                    T=T,
                    gamma=gamma,
                    alpha=alpha,
                    kappa=kappa,
                    state_transition_matrix = state_transition_matrix,
                    observation_generation_matrix = observation_generation_matrix,
                    utility = utility,
                    policies = policies,
                #  prior_rewards = prior_rewards,
                #  counts_prior_rewards = counts_prior_rewards,
                    prior_policies = prior_policies,
                    counts_prior_policies = counts_prior_policies,
                    prior_states = prior_states,
                    na = na,
                    env = env,
                    approx_pred_pol = approx_pred_pol,
                    approx_pred_rew = approx_pred_rew,
                    h = h,
                    debug=debug[rep])



        world = World(agent, env)

        world.simulate_experiment()

        # if agent.K == 2:
        if agent.K > 1:            
            ###

            Q_rew = agent.prior_rewards[-1,:,:,:agent.K]  + 1e-10      # inferred speaker distributions over words
            P_rew = reward_generation_matrix              + 1e-10      # true speaker distributions over words

            labels = np.zeros(2,dtype=int)                  # which learned distribution corresponds to which true distribution 
            true_divergence = np.zeros(2)                   # distance between inferred and true distribution once labels allocated
            divergences = np.zeros((2,agent.K))                   # distance between inferred distribution and all three possible true distributions

            # transition matrix with last row removed
            # tm = (agent.transition_matrix[:agent.K,:agent.K]/agent.transition_matrix[:agent.K,:agent.K].sum(axis=0)[None,:])

            # label allocation
            for distribution in range(2):
                
                for candidate in range(agent.K):
                    
                    divergences[distribution, candidate] = (Q_rew[:,:,candidate]*np.log(Q_rew[:,:,candidate]/P_rew[:,:,distribution])).sum(axis=0).mean()
                
                # the label given is the one with the smallest divergance
                labels[distribution] = np.argmin(divergences[distribution])
                true_divergence[distribution] = divergences[distribution, np.argmin(divergences[distribution])]
                
            ###
                    

            plots = [Q_rew[:2,:2,k] for k in range(agent.K)]
            plots = [Q/Q.sum(axis=0) for Q in plots]

            # plots = [Q_rew[:,:,k] for k in range(agent.K)]
            # titles = [None for k in range(agent.K)]
            # titles[0] = f"alpha: {round(alpha,6)}, gamma: {round(gamma,6)}", f"kappa: {round(kappa,6)}"

            file_title = f"{true_divergence.mean().round(5)}_{agent.K}_{rep}_{i}"

            plot_heatmap(data=plots, file_title=file_title)#, title=titles)

            data = agent
            post_policies = np.nan_to_num(data.posterior_policies[:,:,:,:data.K])
            prior_policies = np.nan_to_num(data.prior_policies[:-1,:,:data.K])
            like_policies = np.nan_to_num(data.likelihood_policies[:,:,:,:data.K])
            post_context = data.posterior_context[:,:,:data.K]
            actions = data.actions


            # plt.style.use('default')

            ### CONTEXT PLOT
            fig, ax = plt.subplots(1, figsize=(5,3))
            ax.set_ylim((0,1.05))
            plt.grid(axis="x")
            ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
            # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
            # plt.hlines(0.5,xmin=0,xmax=switch*2, color = 'k', alpha=0.2)
            for c in range(data.K):
                ax.plot(post_context[:,0,c],label=f"context {c}")
                ax.legend()
                ax.set_title(fr"$\alpha=${alpha}")

            new_context = (data.opened_new_context == True).nonzero()

            for ind in new_context:
                ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5)
            ax.set_title("Posterior Context")
            ax.legend(loc="lower right", framealpha=1)

            plt.show()
            # plt.savefig("test.png",dpi=300)

            # #REWARD ENTROPY PLOT
            # rewards = data.prior_rewards[1:,:,:,:data.K]
            # reward_entropy = np.nan_to_num(-rewards*np.log(rewards)).sum(axis=1).sum(axis=1)/3


            # plt.figure()
            # plt.grid()
            # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
            # for c in range(data.K):
            #     plt.scatter(np.arange(TAU), reward_entropy[:,c], label = f'entropy $c$={c}')
            #     plt.legend()

            # # POLICY PLOT

            # fig, ax = plt.subplots(1)
            # plt.grid()

            # for k in range(data.K):
            #     ax.plot(np.arange(TAU), like_policies[:,0,0,k], '--x',  label = f'like_pol_0_cont_{k}')
            #     ax.plot(np.arange(TAU), like_policies[:,0,1,k], '--x',  label = f'like_pol_1_cont_{k}')
            # # ax.plot(np.arange(TAU), like_policies[:,0,0,1], '-x', label = 'like_pol_0_cont_1')
            # # ax.plot(np.arange(TAU), like_policies[:,0,1,1], '-x',  label = 'like_pol_1_cont_1')
            # ax.set_ylim([0,1])
            # ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
            # plt.legend()
            # ax.set_title("Policy likelihood under the different contexts")

            # ACTION PLOT
            fig, ax = plt.subplots(1)
            plt.grid()
            ax.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
            ax.scatter(np.arange(40), actions[:40,0], marker='x', label = 'action')
            ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
            plt.legend()
            ax.set_title("Chosen action")


            # #### POLICY PLOT
            # post_policies = np.einsum('ktpc,ktc->ktp', post_policies, post_context)
            # prior_policies = np.einsum('ktpc,ktc->ktp', prior_policies[:,None,:,:], post_context)
            # like_policies = np.einsum('ktpc,ktc->ktp', like_policies, post_context)
            # fig, ax = plt.subplots(1)
            # plt.grid()
            # ax.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
            # ax.scatter(np.arange(TAU), post_policies[:,0,1], label = f'posterior $\pi$')
            # ax.scatter(np.arange(TAU), prior_policies[:,0,1], label = f'prior $\pi$')
            # ax.scatter(np.arange(TAU), like_policies[:,0,1], label = f'like $\pi$')
            # ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
            # plt.legend()
            # ax.set_title("Beliefs over policy with context integrated out")

            #### CONTEXT PLOT
            fig, ax = plt.subplots(1)
            plt.grid()
            ax.vlines(switch,ymin=0,ymax=1, color = 'r', linestyle='--', alpha=0.5)
            for ind in new_context:
                ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5)
            ax.scatter(np.arange(TAU), data.context, label = 'inferred context')
            ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
            plt.legend()
            ax.set_title("inferred_context")

            # print(agent.K)
            # for k in range(agent.K):
            #     print(data.prior_rewards[-1,:,:,k].round(3),'\n')

        else:
            true_divergence = np.array([2,2])
        sim_data[i] = np.array([rep, alpha, gamma, kappa, true_divergence.mean().round(5), agent.K])

        if i%500 == 0:
            print(f"{i,rep} alpha: {round(alpha,6)}, gamma: {round(gamma,6)}, kappa: {round(kappa,6)}, K: {agent.K}")
        
        
df = pd.DataFrame(data = sim_data, columns = ["rep","alpha","gamma","kappa","distribution_distance","K"])
df.to_csv("sim_params.csv")

#%%

df = df.dropna()
df = df.loc[~(df==0).all(axis=1)]


for rep in range(reps):

    # Create a 3D scatter plot
    rep_df = df[df["rep"] == rep]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot with color mapping for performance
    scatter = ax.scatter(rep_df['alpha'], rep_df['gamma'], rep_df['kappa'], c=rep_df['distribution_distance'], cmap='viridis', s=40)
    # Add labels
    ax.set_xlabel('alpha')
    ax.set_ylabel('gamma')
    ax.set_zlabel('kappa')
    # Add a colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label('Average DKL[Q_reward||P_reward]')
    plt.show()



for rep in range(reps):

    # Create a 3D scatter plot
    rep_df = df[df["rep"] == rep]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot with color mapping for performance
    scatter = ax.scatter(rep_df['alpha'], rep_df['gamma'], rep_df['kappa'], c=rep_df['K'], cmap='viridis', s=40)
    # Add labels
    ax.set_xlabel('alpha')
    ax.set_ylabel('gamma')
    ax.set_zlabel('kappa')
    # Add a colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label('Average DKL[Q_reward||P_reward]')
    plt.show()


#%%









