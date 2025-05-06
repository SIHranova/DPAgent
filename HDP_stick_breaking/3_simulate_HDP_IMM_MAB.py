#%%
import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp
from itertools import product
from matplotlib.ticker import MultipleLocator
from itertools import product

# from misc import *
from environment import MultiArmedBandit
from agent import HDP,HDP_IMM 
from world import World

plt.rcParams['figure.dpi'] = 100
np.random.seed(8)


def plot_reward_heatmap(data, file_title=str(0), title=None,vmin=0,vmax=1,save=False,dpi=300, rewards=False):
    
    if not type(data) is list:
        data = [data]
        title = [title]


    # fig, axes = plt.subplots(len(data),1, figsize=(3, 1.7*len(data)))
    n_cols = int(np.ceil(len(data)/2)) if len(data)//2 > 0 else 1 
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5,4))
    plt.tight_layout()
    fig.set_dpi(dpi)
    fig.tight_layout()

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ai, ax, im in zip(np.arange(len(data)), axes.flatten(), data):
        g = sns.heatmap(data=im, annot=True, annot_kws={"size":12}, cmap="viridis", cbar=False, fmt='.0f', ax=ax,vmin=vmin, vmax=vmax)

        if title is not None:
            ax.set_title(title[ai])
    
        ax.set_axis_off()
    
    fig.suptitle("Learned reward contingencies", fontsize=14, y=1.06, fontweight="bold" )
    
    if save:
        plt.savefig(file_title + ".png",dpi=300)
        plt.close()
    # return fig, axes


def plot_tm_heatmap(data, file_title=str(0), title=None,vmin=0,vmax=1,save=False,dpi=300, rewards=False):

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # fig, axes = plt.subplots(1,1 // 2, figsize=(5.5,4))
    plt.tight_layout()
    fig.set_dpi(dpi)
    fig.tight_layout()

    g = sns.heatmap(data=data, annot=True, annot_kws={"size":16}, cmap="viridis", cbar=False, fmt='.2f', ax=ax,vmin=vmin, vmax=vmax)
        # g.set_xlabel("states")
        # g.set_ylabel("rewards")

    if title is not None:
        ax.set_title(title)
    
    ax.set_axis_off()
    
    fig.suptitle("Learned context transition matrix", fontsize=14, y=1.06, fontweight="bold" )

    if save:
        plt.savefig(file_title + ".png",dpi=300)
        plt.close()
    # return fig, axes


def plot_conditional_action_probs(df, context_col='context', action_col='action', base_palette='tab10',dpi=100):
    df["context"] += 1
    # Prepare unique contexts and actions
    contexts = sorted(df[context_col].unique())
    actions = sorted(df[action_col].unique())

    # Count and normalize
    counts = df.groupby([context_col, action_col]).size().reset_index(name='count')
    counts['percent'] = counts.groupby(context_col)['count'].transform(lambda x: 100 * x / x.sum())

    # Generate context -> RGB color mapping
    if isinstance(base_palette, str):
        base_colors = sns.color_palette(base_palette, len(contexts))
    else:
        base_colors = base_palette
    context_color_map = dict(zip(contexts, base_colors))

    # Function to generate RGBA with alpha varying by action index
    def get_alpha_color(base_color, idx, total):
        alpha = 1.175 -  0.7 * (idx + 1) / total  # Scale alpha from 0.4 to 1.0
        # print(alpha)
        return (*base_color, alpha)

    # Apply color with alpha per (context, action)
    counts['color'] = counts.apply(
        lambda row: get_alpha_color(
            context_color_map[row[context_col]],
            actions.index(row[action_col]),
            len(actions)
        ),
        axis=1
    )

    # Plotting
    fig, ax = plt.subplots(1, figsize=(2.9, 2),dpi=dpi)
    bar_width = 0.8 / len(actions)
    x = np.arange(len(contexts))

    for i, action in enumerate(actions):
        subset = counts[counts[action_col] == action]
        offsets = x + (i - len(actions)/2) * bar_width + bar_width/2
        ax.bar(
            offsets,
            subset['percent'],
            width=bar_width,
            color=subset['color'],
            edgecolor='black',         # <-- add this
            linewidth=0.8,             # <-- and this (you can tweak thickness)
            label=f'Action {action}'
        )

    # Grayscale legend for actions
    gray_shades = [str(0.1 + 0.7 * (i + 1) / len(actions)) for i in range(len(actions))]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=gray_shades[i])
        for i in range(len(actions))
    ]
    plt.legend(handles, [f"Action {int(a+1)}" for a in actions], bbox_to_anchor=[1,0.85], fontsize=8.5)

    plt.yticks(fontsize=8.5)
    plt.xticks(x, contexts, fontsize=8.5)
    plt.xlabel("Context",fontsize=8.5)
    # plt.ylabel("P(action | context) [%]")
    # plt.title("Conditional Action Probabilities (Grouped)")

    plt.tight_layout()
    plt.show()


def plot_training_regime(training_protocol,nb=4):

    rews = np.array([0.1, 0.9])

    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.xaxis.set_major_locator(MultipleLocator(switch))
    for i in range(nb):
        vals = 0+(training_protocol == i)
        print(rews)
        print(vals)
        plt.plot(np.arange(training_protocol.size), rews[vals])
    plt.ylim(0,1)


# Task setup parameters
na = 4
nb = na
ns = nb+1
no = ns
nr = 3
nc = 1
T = 2
npi = na**(T-1)


plot_rewards = True
plot_transition_matrix = False
plot_context = True
plot_choice = False
debug = False
dpi = 100

switch = 300
repeats = 1
training_protocol = np.tile(np.arange(nb).repeat(switch),repeats)
# plt.rcParams['axes.xaxis.major.locator'] = MultipleLocator(switch)
TAU = training_protocol.size

# plot_training_regime(training_protocol)
# Agent setup Parameters

h = 40
approx_pred_pol = True  # refers to whether digamma is used or not
approx_pred_rew = True
max_context = 6

# I think for these parametrisations worked for 2/3/4 bandits
# for 4 bandits some pretraining was necesary to learn all 4 or many trials! this is at 0.9
# I potentially also played around with the self-transition bias in the extra column as wel.
# gammas = np.array([800])       # global prior context opening tendency
# alphas = np.array([30])        # local  prior context opening tendency
# kappas = np.array([250])       # self-transition bias

# gammas = np.array([3])
# alphas = np.array([2])
# kappas = np.array([4])

gammas = np.array([800])       # global prior context opening tendency
alphas = np.array([30])        # local  prior context opening tendency
kappas = np.array([250])       # self-transition bias


rho_global = np.array([1])     # global prior counts forgetting rate
rho_local = np.array([1])       # local prior counts forgetting rate 


sim_params = product(alphas, gammas, kappas,rho_local, rho_global)
reps = 1   # how many times to run simulation with same params

n_sims = alphas.size*kappas.size*gammas.size*rho_global.size*rho_local.size*reps

sim_data = np.zeros([n_sims,8])

print(f"-----------------------------------")
print(f"{n_sims} simulations to run")
i = -1

###### Run simulations
learned_correct = []

for alpha, gamma, kappa, rho_l, rho_g in sim_params:
    
    for rep in range(reps):

        ####### Setup simulation
        
        i+=1

        '''           define policies            '''
        # policies = list(product(list(np.arange(na))*(T-1)))
        policies = np.array(list(product( np.arange(na), repeat= T-1)))


        '''       define p(s_t|s_t-1, a_t-1)      '''

        prior_states = np.array([0]*nb + [1]) # np.array([0,0,1])
        state_transition_matrix = np.array([np.eye(nb+1)]*nb).transpose([1,2,0])
        state_transition_matrix[:,-1,:] = np.eye(nb+1)[:,:-1]
        
        
        # prior_states = np.array([0,0,1])
        # state_transition_matrix = np.array([[[1,0,1],
        #                                      [0,1,0],
        #                                      [0,0,0]],

        #                                     [[1,0,0],
        #                                      [0,1,1],
        #                                      [0,0,0]]]).transpose(1,2,0)


        '''          define p(o_t|s_t)            '''
        observation_generation_matrix = np.eye(ns)


        '''           define p(r|s,c)             '''
        # lambda_H = np.array([[1,1,1],
        #                      [1,1,1],
        #                      [1,1,100]],dtype=float)
        # counts_prior_rewards = np.stack([lambda_H for i in range(nc)],axis=-1)
        
        lambda_H = np.ones([nr,ns])
        lambda_H[-1,-1] = 100
        counts_prior_rewards = np.stack([lambda_H for i in range(nc)],axis=-1)
        
        bias = 1
        
        init_counts = np.ones([nr,nb+1])
        init_counts[0,0] = bias
        init_counts[1,1:] = bias
        init_counts[:,-1] = [1,1,100]
        counts_prior_rewards[:,:,0] = init_counts
        
        # counts_prior_rewards[:,:,0] = np.array([[bias, 1   , 1   ],
        #                                         [1   , bias, 1   ],
        #                                         [1   , 1   , 100]])
        
        # counts_prior_rewards[:,:,0] +=  np.random.uniform(low=0,high=1,size=(nr,ns))*0.3

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
        utility = np.array([0.99, 0.005,0.005]) # np.array([1/nr]*3) #


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

        
        # reward_generation_matrix =  np.array([[[p, q, 0], 
        #                                        [q, p, 0], 
        #                                        [0, 0, 1]], 
                                            
        #                                       [[q, p, 0], 
        #                                        [p, q, 0], 
        #                                        [0, 0, 1]]]).transpose(1,2,0)


        ######## Plot task setup
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

        ##### Setup Classes
        env = MultiArmedBandit(state_transition_matrix,
                            reward_generation_matrix, 
                            TAU=TAU,
                            T=T,
                            n_bandits = nb,
                            training_protocol=training_protocol,
                            observation_generation_matrix=observation_generation_matrix,
                            no=no)


        agent = HDP_IMM(lambda_H = lambda_H,
                    TAU=TAU,
                    T=T,
                    gamma=gamma,
                    alpha=alpha,
                    kappa=kappa,
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
                    max_context=max_context,
                    K = nc)



        world = World(agent, env)
        world.simulate_experiment()




        ############### 
        if agent.K >= 1:  # K is number of inferrred contexts
            

            ### FOR SARAH: this part of the code deals with how close inferred distributions are to true reward distributions
            ### It is also kind of broken so you can ignore it
            Q_rew = agent.prior_rewards_counts[-1,:,:,:agent.K]  + 1e-10      # inferred reward distribution given state and context
            P_rew = reward_generation_matrix              + 1e-10      # true reward distribution

            labels = np.zeros(2,dtype=int)                             # which learned distribution corresponds to which true distribution 
            true_divergence = np.zeros(2)                              # distance between inferred and true distribution once labels allocated
            divergences = np.zeros((2,agent.K))                        # distance between inferred distribution and all three possible true distributions

            # Label allocation
            for distribution in range(2):
                
                for candidate in range(agent.K):
                    
                    divergences[distribution, candidate] = (Q_rew[:,:,candidate]*np.log(Q_rew[:,:,candidate]/P_rew[:,:,distribution])).sum(axis=0).mean()
                
                # the label given is the one with the smallest divergance
                labels[distribution] = np.argmin(divergences[distribution])
                true_divergence[distribution] = divergences[distribution, np.argmin(divergences[distribution])]
            ###
            best_fit = 0
            best_label = []        
            for perm in range(100):    
                l = np.random.permutation(agent.K)        
                labels = l[agent.context]
                fit = (labels == training_protocol).sum()/TAU
                
                if fit > best_fit:
                    best_fit = fit
                    best_label = l
        
            # print(f"best label assignment: {best_label}")
           
            Q_rew = agent.prior_rewards_counts[-1,:,:,:agent.K]  + 1e-10      # inferred reward distribution given state and context
            plots = [(Q_rew[:,:,k]).round().astype(int) for k in range(agent.K)]
            titles = [f"Context {k+1}" for k in range(agent.K)]
            # titles[0] = f"l: {best_label}, alpha: {round(alpha,6)}, gamma: {round(gamma,6)}", f"kappa: {round(kappa,6)}"

            file_title = f"{true_divergence.mean().round(5)}_{agent.K}_{rep}_{i}"

            ### plots inferred reward distributions for each context
            if plot_rewards:

                # plots = [plot[:nr-1,:nb] / plot[:nr-1,:nb].sum(axis=0)[None,:]  for plot in plots]
                plot_reward_heatmap(data=plots, file_title=file_title, title=titles, dpi=dpi,rewards=True,vmax=None)
            if plot_transition_matrix:
                plot_tm_heatmap(agent.transition_matrix[:agent.K+1,:agent.K+1].round(2),dpi=dpi)


            ### CONTEXT PLOT
            
            data = agent
            post_policies = np.nan_to_num(data.posterior_policies[:,:,:,:data.K])
            prior_policies = np.nan_to_num(data.prior_policies[:-1,:,:data.K])
            like_policies = np.nan_to_num(data.likelihood_policies[:,:,:,:data.K])
            post_context = data.posterior_context[:,:,:data.K+1]
            actions = data.actions
            
            if plot_context:
                # unexpected_event = np.zeros(TAU)
                
                # for trial, trial_type in enumerate(training_protocol):
                #     if trial_type == 0:
                #         unexpected_event[trial] = agent.observations[trial,1] != agent.rewards[trial,1] 
                #     else:
                #         unexpected_event[trial] = agent.observations[trial,1] == agent.rewards[trial,1] 

                # inf_context = np.argmax(agent.posterior_context[:,1,:],axis=1)
                # y_val_unexp_event = agent.posterior_context[np.arange(TAU),1,inf_context]*unexpected_event
                # y_val_unexp_event[y_val_unexp_event == 0] = None

                # y_val_action = agent.posterior_context[np.arange(TAU),1,inf_context]*agent.actions[:,0]
                # y_val_action[y_val_action == 0] = None

                fig, ax = plt.subplots(1, figsize=(3,3), dpi=dpi)

                K = K = np.cumsum(agent.opened_new_context)+1 
                novel_context = data.posterior_context[np.arange(TAU),:,K[:-1]]
                post_context[np.arange(TAU),:,K[:-1]] = 0

                ax.set_ylim((0,1.05))
                plt.grid(axis="x", alpha=0.7)
                # plt.grid(axis="y", alpha=0.7)
                # plt.grid(axis="y")
                ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
                # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
                # plt.hlines(0.5,xmin=0,xmax=switch*2, color = 'k', alpha=0.2)
                # ax.scatter(np.arange(TAU), y_val_action, marker="o", color="r", s=30)
                # ax.scatter(np.arange(TAU), y_val_unexp_event, marker="x", color="k")
                post_context /= post_context.sum(axis=-1)[:,:,None]

                for c in range(data.K):
                    ax.plot(post_context[:,1,c],label=f"Context {c+1}", linewidth=1.5)
                
                ax.plot(novel_context[:,1], 'gray', label=f"Template\ncontext")
                
                
                new_context = (data.opened_new_context == True).nonzero()
                
                for i, ind in enumerate(new_context):
                    if i == len(new_context)-1:
                        ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5, label="Context\nopened")    
                    else:
                        ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5)
                
                ax.set_title(f"Posterior Context")#, correct in {(100*best_fit).round()}% of trials", fontsize=14, y = 1.05)
                ax.legend(bbox_to_anchor=[1.05,1.05], framealpha=1, labelspacing = 1, fontsize=14)
                # ax.legend(bbox_to_anchor=[2,-0.22], framealpha=1, fontsize=14, ncols = agent.K+2)

                ax.set_xlabel("trial", fontsize=14)
                ax.tick_params(labelsize=14)#, rotation = 45)

            print(f"rep: {best_fit.round(3)}")
            learned_correct.append(best_fit)
            if plot_choice:
                df = pd.DataFrame({"trial": np.arange(TAU), "action": agent.actions[:,0], "context":training_protocol})
                plot_conditional_action_probs(df,dpi=dpi)

            ##### messages plot?
            # counts = agent.global_prior_counts.copy()
            # q_z = np.array([agent.digamma_approximation(counts[trial]) for trial  in range(TAU)])
            # obs_messages = np.nan_to_num(agent.context_likelihood)


            # result = q_z*obs_messages
            # result[result == 0] = -1000
            # result = np.argmax((result),axis=1)

            # q_z[q_z == 0 ] = -1000
            # z = np.argmax(q_z,axis=1)
            # obs_messages[obs_messages==0] = -1000
            # obs = np.argmax(obs_messages,axis=1)
            # q_z[q_z == -1000 ] = None
            # obs_messages[obs_messages == -1000] = 0

            # nc_plot = 6
            # plt.figure()
            # plt.grid()
            # ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
            # plt.plot(np.arange(TAU-1), obs_messages[1:,:nc_plot],label=[f"obs {c}" for c in range(nc_plot)])
            # plt.plot(np.arange(TAU-1), (q_z*obs_messages)[1:,:nc_plot],'-x',label = [f"q_z*obs {c}" for c in np.arange(nc_plot)])
            # plt.plot(np.arange(TAU-1), q_z[1:,:nc_plot],'--', label = [f"q_z {c}" for c in np.arange(nc_plot)])
            # # plt.ylim([-8,1.5])
            # plt.legend(bbox_to_anchor=[1,0.8])


            # plt.show()
        else:
            best_fit = 0
            true_divergence = np.array([1000,1000]) # just set to somethin high
        
        sim_data[i] = np.array([rep, alpha, gamma, kappa, rho_l, true_divergence.mean().round(5), agent.K, best_fit])

        if i%500 == 0:
            print(f"{i,rep} alpha: {round(alpha,6)}, gamma: {round(gamma,6)}, kappa: {round(kappa,6)}, rho: {round(rho_l,6)}, K: {agent.K}")


print(f"\n\n total: {(np.array(learned_correct) > 0.8).sum()/n_sims}")
# plt.show()

        




#%%%
%matplotlib widget
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ─── DATA SETUP ────────────────────────────────────────────────────────────────
n_frames = 50
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ─── DATA SETUP ────────────────────────────────────────────────────────────────

def plot_reward_heatmap(data, file_title=str(0), title=None,vmin=0,vmax=1,save=False,dpi=300, rewards=False):
    
    if not type(data) is list:
        data = [data]
        title = [title]


    # fig, axes = plt.subplots(len(data),1, figsize=(3, 1.7*len(data)))
    n_cols = int(np.ceil(len(data)/2)) if len(data)//2 > 0 else 1 
    fig, axes = plt.subplots(2, n_cols, figsize=(5.5,4))
    plt.tight_layout()
    fig.set_dpi(dpi)
    fig.tight_layout()

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ai, ax, im in zip(np.arange(len(data)), axes.flatten(), data):
        g = sns.heatmap(data=im, annot=True, annot_kws={"size":12}, cmap="viridis", cbar=False, fmt='.0f', ax=ax,vmin=vmin, vmax=vmax)

        if title is not None:
            ax.set_title(title[ai])
    
        ax.set_axis_off()
    
    fig.suptitle("Learned reward contingencies", fontsize=14, y=1.06, fontweight="bold" )
    
    if save:
        plt.savefig(file_title + ".png",dpi=300)
        plt.close()
    # return fig, axes

n_frames = 10
heatmap_size = 10
# heatmap_data[f, i] is the (heatmap_size × heatmap_size) array for subplot i at frame f
rewards_data = agent.prior_rewards_counts[:10,:,:,:agent.K].transpose(0,3,1,2)
heatmap_data = rewards_data[0]
# rewards_data = rewards_data[1:]
# line_data[f] is the y-array (of length heatmap_size) at frame f
x = np.linspace(0, 10, heatmap_size)
line_data = np.sin(np.outer(np.linspace(0, 2 * np.pi, n_frames), x))  # example: moving sine wave

# ─── FIGURE 1: FOUR HEATMAPS ───────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

# Draw initial heatmaps and grab the QuadMesh artists so we can update them
heatmap_artists = []
for idx, ax in enumerate(axes):
    sns.heatmap(
        heatmap_data[idx,:,:], ax=ax, annot=True, annot_kws={"size":12}, cmap="viridis", cbar=False, fmt='.0f'
    )
    ax.set_title(f"Heatmap {idx+1}")
    artist = ax.collections[0]              # the QuadMesh created by seaborn
    heatmap_artists.append(artist)

plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom for slider/buttons

# ─── FIGURE 2: LINE PLOT ───────────────────────────────────────────────────────
fig2, ax_line = plt.subplots(figsize=(6, 4))
sns.lineplot(x=x, y=line_data[0], ax=ax_line)
line_artist = ax_line.lines[0]
ax_line.set_xlim(x.min(), x.max())
ax_line.set_ylim(line_data.min(), line_data.max())
ax_line.set_title("Line Plot")

# ─── WIDGETS ───────────────────────────────────────────────────────────────────
# Slider
slider_ax = fig1.add_axes([0.25, 0.03, 0.5, 0.03])
frame_slider = Slider(
    ax=slider_ax, label="Frame", valmin=0, valmax=n_frames-1,
    valinit=0, valstep=1, color='lightblue'
)

# Buttons
btn_ax_prev = fig1.add_axes([0.05, 0.75, 0.1, 0.04])
btn_ax_next = fig1.add_axes([0.05, 0.68, 0.1, 0.04])
btn_ax_play = fig1.add_axes([0.05, 0.61, 0.1, 0.04])
btn_ax_stop = fig1.add_axes([0.05, 0.54, 0.1, 0.04])

btn_prev = Button(btn_ax_prev, 'Previous')
btn_next = Button(btn_ax_next, 'Next')
btn_play = Button(btn_ax_play, 'Play')
btn_stop = Button(btn_ax_stop, 'Stop')

# ─── UPDATE CALLBACK ───────────────────────────────────────────────────────────
def update(frame):
    """Update heatmaps and line plot to the given frame index."""
    frame = int(frame)
    # update each heatmap
    for idx, artist in enumerate(heatmap_artists):
        new_data = rewards_data[frame,idx,:,:]
        print(new_data)
        artist.set_array(new_data.ravel())
    # update line
    line_artist.set_ydata(line_data[frame])
    # redraw
    fig1.canvas.draw_idle()
    fig2.canvas.draw_idle()

frame_slider.on_changed(update)

# ─── PLAY/PAUSE TIMER ──────────────────────────────────────────────────────────
timer = fig1.canvas.new_timer(interval=200)  # interval in milliseconds

def _advance():
    """Advance the slider by one, looping at the end."""
    next_val = (int(frame_slider.val) + 1) % n_frames
    frame_slider.set_val(next_val)

timer.add_callback(_advance)

btn_play.on_clicked(lambda event: timer.start())
btn_stop.on_clicked(lambda event: timer.stop())

# ─── NEXT / PREVIOUS BUTTONS ──────────────────────────────────────────────────
btn_next.on_clicked(lambda event: frame_slider.set_val(
    min(frame_slider.val + 1, n_frames - 1)
))
btn_prev.on_clicked(lambda event: frame_slider.set_val(
    max(frame_slider.val - 1, 0)
))

# ─── SHOW EVERYTHING ───────────────────────────────────────────────────────────
plt.show()

