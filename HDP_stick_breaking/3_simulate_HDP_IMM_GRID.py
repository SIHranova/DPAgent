#%%
import numpy as np
np.set_printoptions(suppress=True)
# %matplotlib widget
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp
from itertools import product
from matplotlib.ticker import MultipleLocator
from itertools import product
# from misc import *
from environment import GridWorld
from agent import HDP,HDP_IMM, HDP_correct
from world import World
    
plt.rcParams['figure.dpi'] = 100
np.random.seed(2)




def plot_rewards_heatmap(data, file_title=str(0), title=None,vmin=0,vmax=1,save=False,dpi=200, rewards=False,fmt='.2f'):
    
    if not type(data) is list:
        data = [data]
        title = [title]

    fig, axes = plt.subplots(len(data),1, figsize=(6, len(data)*1.1))
    plt.tight_layout()
    fig.set_dpi(dpi)
    fig.tight_layout()
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])


    for ai, ax, im in zip(np.arange(len(data)), axes.flatten(), data):
        g = sns.heatmap(data=im, annot=True, annot_kws={"size":6}, cmap="viridis", cbar=False, fmt=fmt, ax=ax,vmin=vmin, vmax=vmax)
        # g.set_xlabel("states")
        # g.set_ylabel("rewards")

        if title is not None:
            ax.set_title(title[ai])
    
        ax.set_axis_off()

    fig.suptitle("Learned reward contingencies", fontsize=6, y=1.06, fontweight="bold" )

  
    if save:
        plt.savefig(file_title + ".png",dpi=300)
        plt.close()

    return fig, axes


# Task setup parameter
plot_rewards = False
plot_transition_matrix = False
plot_context = True
plot_context_obs = False
plot_choice = False
plot_messages = False

plot_avg_context_posterior = False
plot_avg_context_accuracy= False
plot_avg_context_entropy_and_accuracy = False
use_context_obs = False
use_template = True
debug = False
dpi = 100


switch = [100]#[100,100,300]
repeats = 2

approx_pred_pol = True  # refers to whether digamma is used or not
approx_pred_rew = True


# # I think for these parametrisations worked for 2/3/4 bandits
# # for 4 bandits some pretraining was necesary to learn all 4 or many trials! this is at 0.9
# # I potentially also played around with the self-transition bias in the extra column as wel.

gammas = np.array([850])       # global prior context opening tendency
alphas = np.array([30])        # local  prior context opening tendency
kappas = np.array([250])       # self-transition bias
hs =    np.array([30])         # np.floor(np.exp(np.arange(1,9.5,0.25)))  # np.array([10000])    #
rho_global = np.array([1])     # global prior counts forgetting rate
rho_local = np.array([1])      # local prior counts forgetting rate 
state_unc = False
gamma_init = 1000
cap = 100000

# gammas = np.array([28])       # global prior context opening tendency
# alphas = np.array([1])        # local  prior context opening tendency
# kappas = np.array([40])       # self-transition bias
# hs =    np.array([10000000])         # np.floor(np.exp(np.arange(1,9.5,0.25)))  # np.array([10000])    #
# rho_global = np.array([1])     # global prior counts forgetting rate
# rho_local = np.array([1])      # local prior counts forgetting rate 
# state_unc = False
# gamma_init = 30
# cap = 100000



sim_params = product(alphas, gammas, kappas, hs, rho_local, rho_global)
reps = 2 # how many times to run simulation with same params

n_sims = alphas.size*kappas.size*gammas.size*hs.size*rho_global.size*rho_local.size*reps

print(f"-----------------------------------")
print(f"{n_sims} simulations to run")
i = -1

###### Run simulations
learned_correct = []

dfs = []

for alpha, gamma, kappa, h, rho_l, rho_g in sim_params:  
    for rep in range(reps):


        T = 5 #number of time steps in each trial
        Lx = 4 #grid length
        Ly = 5
        no = Lx*Ly #number of observations
        ns = Lx*Ly #number of states
        na = 3 #number of actions
        npi = na**(T-1)
        nr = 2
        nc = 0
        actions = np.array([[0,-1], [1,0], [0,1]])
        start = 2
        max_context = 7
        
        goal = np.array([14,10])
        nco = goal.size
        training_protocol = np.tile(goal.repeat(switch[0]), repeats)
        TAU = training_protocol.size

        u = 0.999
        utility = np.array([1-u,u])


        '''           define policies            '''
        policies = np.array(list(product(list(range(na)), repeat=T-1)))
        npi = policies.shape[0]


        '''       define p(s_t|s_t-1, a_t-1)      '''
        prior_states = np.zeros((ns))
        prior_states[start] = 1

        vals = np.array([1., 2/3., 1/2., 1./2.])
        const = 0

        state_transition_matrix = np.zeros((ns, ns, na)) + const
            
        cert_arr = np.zeros(ns)

        for s in range(ns):
            x = s//Ly
            y = s%Ly

            #state uncertainty condition
            if state_unc:
                if (x==0) or (y==3):
                    c = vals[0]
                elif (x==1) or (y==2):
                    c = vals[1]
                elif (x==2) or (y==1):
                    c = vals[2]
                else:
                    c = vals[3]

                condition = 'state'

            else:
                c = 1.

            cert_arr[s] = c
            for u in range(na):
                x = s//Ly+actions[u][0]
                y = s%Ly+actions[u][1]

                #check if state goes over boundary
                if x < 0:
                    x = 0
                elif x == Lx:
                    x = Lx-1

                if y < 0:
                    y = 0
                elif y == Ly:
                    y = Ly-1

                s_new = Ly*x + y
                if s_new == s:
                    state_transition_matrix[s, s, u] = 1 - (ns-1)*const
                else:
                    state_transition_matrix[s, s, u] = 1-c + const
                    state_transition_matrix[s_new, s, u] = c - (ns-1)*const


        '''          define p(o_t|s_t)            '''
        observation_generation_matrix = np.eye(ns)


        '''           define p(r|s,c)             '''
        lambda_H = np.ones([nr,ns])
        lambda_H[0,:] = 1
        # lambda_H[0,start] = 10
        init_counts = np.ones([nr,ns])
        # init_counts[0,start] = 10
        counts_prior_rewards = np.zeros([nr,ns,nc])#np.stack([lambda_H for i in range(nc)],axis=-1)

        for c in range(nc):
            counts_prior_rewards[:,:,c] = init_counts
        counts_prior_rewards[0,:,:] = 1


        template_context_contingencies = np.ones([nr,ns,ns])
        bias = 10
        for s in range(0,ns):
            template_context_contingencies[0,np.where(np.arange(ns) != s),s] = 10 
            template_context_contingencies[1,s,s] = 10






        if approx_pred_rew:
            prior_rewards = scp.digamma(counts_prior_rewards) - scp.digamma(counts_prior_rewards.sum(axis=0))
            prior_rewards = scp.softmax(prior_rewards, axis=0)
        else:
            prior_rewards = counts_prior_rewards / counts_prior_rewards.sum(axis=0)
      


        '''   define Env reward generation matrix '''
        reward_generation_matrix = np.zeros((nr,ns, ns))
        for s in range(ns):
            reward_generation_matrix[0,np.where(np.arange(ns) != s),s] = 1
            reward_generation_matrix[1,s,s] = 1

        reward_generation_matrix[reward_generation_matrix == 0] = 0


        '''          define p(d|c)                '''
        p = 0.9
        q = 1-p
        # context_obs_generation_matrix = np.ones([nco, na])*(q/(na-1))
        # context_obs_generation_matrix[np.arange(nco), np.arange(nco)] = p
        context_obs_generation_matrix = np.zeros((nco, ns)) + 1/nco

        '''           define counts alpha in p(pi|theta,alpha)           '''
        counts_prior_policies = np.zeros([npi,nc+1]) + h
        
        if approx_pred_pol:
            prior_policies = scp.softmax(scp.digamma(counts_prior_policies) - scp.digamma(counts_prior_policies.sum(axis=0)))
        else:
            prior_policies = counts_prior_policies / counts_prior_policies.sum(axis=0)


        '''       define dummy utility RV p(R=1) '''
        p = 0.999
        utility = np.array([1-p,p]) # np.array([1/nr]*3) #


        '''   define context obs contingencies '''
        context_observation_counts = np.ones([nco,nc+1])


        '''    define classses'''    
        env = GridWorld(observation_generation_matrix,
                        state_transition_matrix,
                        reward_generation_matrix,
                        training_protocol=training_protocol,
                        TAU=TAU,
                        T=T, initial_state=2,
                        context_observation_generation_matrix = context_obs_generation_matrix,
                        no=no,nr=nr)


        agent = HDP_correct(lambda_H = lambda_H,
                    TAU=TAU,
                    T=T,
                    gamma=gamma,
                    alpha=alpha,
                    kappa=kappa,
                    K = nc,
                    max_context=max_context,
                    state_transition_matrix = state_transition_matrix,
                    observation_generation_matrix = observation_generation_matrix,
                    utility = utility,
                    policies = policies,
                    # prior_rewards = prior_rewards,
                    counts_prior_rewards = counts_prior_rewards,
                    prior_policies = prior_policies,
                    counts_prior_policies = counts_prior_policies,
                    prior_states = prior_states,
                    na = na,
                    env = env,
                    approx_pred_pol = approx_pred_pol,
                    approx_pred_rew = approx_pred_rew,
                    h = h,
                    debug=debug, # If set to True will print inferred agent beliefs up to trial 40?
                    rho_l= rho_l,
                    rho_g = rho_g,
                    template_context_contingencies = template_context_contingencies,
                    context_observation_counts=context_observation_counts,
                    use_context_obs=use_context_obs,
                    gamma_init = gamma_init,
                    cap=cap,
                    use_template = use_template)


        world = World(agent, env, training_protocol=training_protocol)
        world.simulate_experiment()



        agent = world.agent

        if plot_rewards:
            plot_rewards_heatmap([agent.prior_rewards_counts[-1,:,:,k] for k in range(world.agent.K+1)],vmax=None)

        ### CONTEXT PLOT
        agent = agent
        post_context = agent.posterior_context[:,:,:].copy()
       
        if plot_context:
            K = np.cumsum(agent.opened_new_context)+nc 
            novel_context = agent.posterior_context[np.arange(TAU),:,K[:-1]]
            post_context[np.arange(TAU),:,K[:-1]] = 0
            post_context /= post_context.sum(axis=-1)[:,:,None]

            new_context = (agent.opened_new_context == True).nonzero()
                
            if plot_context:
                fig, ax = plt.subplots(1, figsize=(3,3), dpi=dpi)
                ax.set_ylim((0,1.05))
                plt.grid(axis="x", alpha=0.7)
                ax.xaxis.set_major_locator(MultipleLocator(switch[0]))  # Set tick spacing on x-axis to 1
                # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
                # plt.hlines(0.5,xmin=0,xmax=switch*2, color = 'k', alpha=0.2)
                # ax.scatter(np.arange(TAU), y_val_action, marker="o", color="r", s=30)
                # ax.scatter(np.arange(TAU), y_val_unexp_event, marker="x", color="k")

                for c in range(agent.K):
                    ax.plot(post_context[:,-1,c],label=f"Context {c+1}", linewidth=1)

                ax.plot(novel_context[:,-1], 'gray', label=f"Novel\ncontext")


                for i, ind in enumerate(new_context):
                    if i == len(new_context)-1:
                        ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5, label="Context\nopened")    
                    else:
                        ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5)

                # ax.set_title(f"{gamma}; Posterior Context; correct in {(100*best_fit).round()}% of trials", fontsize=14, y = 1.05)
                ax.legend(bbox_to_anchor=[1.05,1.05], framealpha=1, labelspacing = 1, fontsize=14)
                # ax.legend(bbox_to_anchor=[2,-0.22], framealpha=1, fontsize=14, ncols = agent.K+2)
                ax.set_ylabel("Posterior Context", fontsize=14)
                ax.set_xlabel("trial", fontsize=14)
                ax.tick_params(axis="x", labelsize=12, rotation = 40)

                plt.show()


        # if True:
        #     fig, ax = plt.subplots(1, figsize=(3,3), dpi=dpi)
        #     ax.set_ylim((0,1.05))
        #     plt.grid(axis="x", alpha=0.7)
        #     ax.xaxis.set_major_locator(MultipleLocator(switch[0]))  # Set tick spacing on x-axis to 1
        #     # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
        #     # plt.hlines(0.5,xmin=0,xmax=switch*2, color = 'k', alpha=0.2)
        #     # ax.scatter(np.arange(TAU), y_val_action, marker="o", color="r", s=30)
        #     # ax.scatter(np.arange(TAU), y_val_unexp_event, marker="x", color="k")

        #     for c in range(agent.K):
        #         ax.plot(agent.post_context[:,-1,c],label=f"Context {c+1}", linewidth=1)

        #     ax.plot(novel_context[:,1], 'gray', label=f"Novel\ncontext")
        