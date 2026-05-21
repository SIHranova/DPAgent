#%%
import numpy as np
# np.set_printoptions(suppress=True)
# %matplotlib widget
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp
from itertools import product
from matplotlib.ticker import MultipleLocator

# from misc import *
from environment import MultiArmedBandit
from agent import HDP, HDP_correct
from world import World

plt.rcParams['figure.dpi'] = 100
np.random.seed(1)


def plot_rewards_heatmap(data, file_title=str(0), title=None,vmin=0,vmax=1,save=False,dpi=300, rewards=False,fmt='.2f'):
    
    if not type(data) is list:
        data = [data]
        title = [title]

    # fig, axes = plt.subplots(len(data),1, figsize=(3, 1.7*len(data)))
    n_cols = len(data) // 2 + len(data) %  2
    fig, axes = plt.subplots(2, len(data) // 2 + len(data) %  2 , figsize=(n_cols*3,4))
    plt.tight_layout()
    fig.set_dpi(dpi)
    fig.tight_layout()
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])


    for ai, ax, im in zip(np.arange(len(data)), axes.flatten(), data):
        g = sns.heatmap(data=im, annot=True, annot_kws={"size":16}, cmap="viridis", cbar=False, fmt=fmt, ax=ax,vmin=vmin, vmax=vmax)
        # g.set_xlabel("states")
        # g.set_ylabel("rewards")

        if title is not None:
            ax.set_title(title[ai])
    
        ax.set_axis_off()

    fig.suptitle("Learned reward contingencies", fontsize=14, y=1.06, fontweight="bold" )

  
    if save:
        plt.savefig(file_title + ".png",dpi=300)
        plt.close()


def plot_transition_matrix_heatmap(data, file_title=str(0), title=None,vmin=0,vmax=1,save=False,dpi=300, rewards=False):
    
    if not type(data) is list:
        data = [data]
        title = [title]
    
    fig, axes = plt.subplots(1, len(data), figsize=(4, 4*len(data)))
    # fig, axes = plt.subplots(1,1 // 2, figsize=(5.5,4))
    plt.tight_layout()
    fig.set_dpi(dpi)
    fig.tight_layout()
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    

    for ai, ax, im in zip(np.arange(len(data)), axes.flatten(), data):
        g = sns.heatmap(data=im, annot=True, annot_kws={"size":16}, cmap="viridis", cbar=False, fmt='.2f', ax=ax,vmin=vmin, vmax=vmax)
        # g.set_xlabel("states")
        # g.set_ylabel("rewards")

        if title is not None:
            ax.set_title(title[ai])
    
        ax.set_axis_off()
    
    fig.suptitle("Learned context transition matrix", fontsize=14, y=1.06, fontweight="bold" )

    if save:
        plt.savefig(file_title + ".png",dpi=300)
        plt.close()


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

# Task setup parameters

plot_rewards = True
plot_transition_matrix = False
plot_context = True
plot_context_obs = False
plot_choice = False 
plot_messages = False


plot_example_rewards = False
plot_avg_context_posterior = False
plot_avg_context_accuracy= False
plot_avg_context_entropy_and_accuracy = False
use_context_obs = False
use_template = False
debug = False
dpi = 100

repeats = 1

approx_pred_pol = True  # refers to whether digamma is used or not
approx_pred_rew = True


number_of_bandits = np.array([2])



gammas = np.array([1])#,2,3,4,5,6])       # global prior context opening tendency
switch = np.array([100])
alphas = np.array([1])#,2,3,4,5])         # local  prior context opening tendency
kappas = np.array([1])#,2,3,4,5,6,7,8,9,10])        # self-transition bias
hs = np.array([10000])
rho_global = np.array([1])     # global prior counts forgetting rate
rho_local = np.array([1])

gamma_init = 1
cap = 1000000

# gammas = np.array([850])       # global prior context opening tendency
# switch = np.array([100])
# alphas = np.array([30])         # local  prior context opening tendency
# kappas = np.array([250])        # self-transition bias
# hs = np.array([10000])
# rho_global = np.array([1])     # global prior counts forgetting rate
# rho_local = np.array([1])

# gamma_init = 1000
# cap = 1000000

sim_params = product(alphas, gammas, kappas, hs, rho_local, rho_global, number_of_bandits)
reps = 1 # how many times to run simulation with same params

n_sims = alphas.size*kappas.size*gammas.size*hs.size*rho_global.size*rho_local.size*number_of_bandits.size*reps

i = -1

###### Run simulations
learned_correct = []

dfs = []

for alpha, gamma, kappa, h, rho_l, rho_g, na in sim_params:  
    for rep in range(reps):

        nb = na
        ns = nb+1
        no = ns
        nco = na
        nr = 3
        nc = 0
        nt = na
        max_context = 6
        T = 2
        npi = na**(T-1)
        ind = np.arange(number_of_bandits.size)[number_of_bandits == nb][0]
        training_protocol = np.tile(np.arange(nb).repeat(switch[ind]),repeats)
        TAU = training_protocol.size

        ####### Setup simulation
        
        i+=1

        '''           define policies            '''
        policies = np.array(list(product( np.arange(na), repeat= T-1)))


        '''       define p(s_t|s_t-1, a_t-1)      '''

        prior_states = np.array([0]*nb + [1]) # np.array([0,0,1])
        state_transition_matrix = np.array([np.eye(nb+1)]*nb).transpose([1,2,0])
        state_transition_matrix[:,-1,:] = np.eye(nb+1)[:,:-1]


        '''          define p(o_t|s_t)            '''
        observation_generation_matrix = np.eye(ns)


        '''           define p(r|s,c)             '''
        
        lambda_H = np.ones([nr,ns])
        lambda_H[-1,-1] = 100
        counts_prior_rewards = np.zeros([nr,nb+1,nc])#np.stack([lambda_H for i in range(nc)],axis=-1)
        
        bias = 1
        for c in range(nc):
            init_counts = np.ones([nr,nb+1])
            init_counts[1,c] = bias
            init_counts[0,np.arange(na+1) != c] = bias
            init_counts[:,-1] = [1,1,100]
            counts_prior_rewards[:,:,c] = init_counts
            # counts_prior_rewards[:,:,c] += np.random.uniform(low=0, high=1, size=(nr,ns))
            
        # IMPLEMENT CREATION OF TEMPLATE CONTEXTS
        template_context_contingencies = np.ones([nr,nb+1,na])
        bias = 10
        for temp in range(0,nt):
            template_context_contingencies[1,temp,temp] = bias
            template_context_contingencies[0, np.arange(na+1) != temp, temp] = bias
            template_context_contingencies[:,-1,:] = np.array([1,1,100])[:,None]
        template_context_contingencies += np.random.uniform(low=0, high=1, size=(nr,ns,nt))


        if approx_pred_rew:
            prior_rewards = scp.digamma(counts_prior_rewards) - scp.digamma(counts_prior_rewards.sum(axis=0))
            prior_rewards = scp.softmax(prior_rewards, axis=0)
        else:
            prior_rewards = counts_prior_rewards / counts_prior_rewards.sum(axis=0)



        '''   define Env reward generation matrix '''
        p = 0.9
        q = 1 - p
        bandits = np.arange(nb)
        reward_generation_matrix = np.ones([nr,ns,len(bandits)])*q

        for context,b in enumerate(bandits):
            reward_generation_matrix[1,b,context] = p
            reward_generation_matrix[0,np.arange(nb+1) != b,context] = p

        reward_generation_matrix[-1,:] = 0
        reward_generation_matrix[:,-1] = 0
        reward_generation_matrix[-1,-1] = 1
        '''          define p(d|c)                '''
        p = 0.9
        q = 1-p
        context_obs_generation_matrix = np.ones([nco, na])*(q/(na-1))
        context_obs_generation_matrix[np.arange(nco), np.arange(nco)] = p
        '''           define counts alpha in p(pi|theta,alpha)           '''
        counts_prior_policies = np.zeros([npi,nc+1]) + h

        if approx_pred_pol:
            prior_policies = scp.softmax(scp.digamma(counts_prior_policies) - scp.digamma(counts_prior_policies.sum(axis=0)))
        else:
            prior_policies = counts_prior_policies / counts_prior_policies.sum(axis=0)


        '''       define dummy utility RV p(R=1) '''
        utility = np.array([0.005, 0.99, 0.005])

        '''   define context obs contingencies '''
        context_observation_counts = np.ones([nco,nc+1])



        ######## Plot task setup
        if False:
            '''Plot state transition matrix'''
            fig,axes = plt.subplots(1,na,figsize=(na*3,3))
            
            fig.suptitle("$p(s_{t}|s_{t-1},a)$",y=1.2, fontsize=14)
            for ai, ax,title in zip(np.arange(na), axes, [rf'$a_{i}$ = L{i}' for i in range(1,na+1)]):
            # for ai, ax,title in zip([0,1], axes, ['$a_1$ = L1', '$a_2$ = L2']):
                ax.xaxis.tick_top()
                sns.heatmap(state_transition_matrix[:,:,ai],annot=True,ax=ax, cbar=False, cmap="viridis")
                ax.set_xticklabels([f'L{i}' for i in range(1,na+1)] + ['init'],minor=False);
                ax.set_yticklabels([f'L{i}' for i in range(1,na+1)] + ['init'],minor=False);
                ax.set_title(title);

            '''Plot likelihood matrix'''
            fig, axes = plt.subplots(1,nc, figsize=(nc*3,3))
            
            if nc == 1:
                axes = sns.heatmap(prior_rewards[...,0],cbar=False,annot=True, cmap="viridis")
                axes.set_xticklabels([ f'L{i}' for i in range(1,na+1)] + ['init'],minor=False)
                axes.set_yticklabels([f'$r_{r}$' for r in range(nr)])

            else:
                for ai, ax in enumerate(axes):
                    ax = sns.heatmap(prior_rewards[...,ai],cbar=False,annot=True, cmap="viridis")
                    ax.set_xticklabels([ f'L{i}' for i in range(1,na+1)] + ['init'],minor=False)
                    ax.set_yticklabels([f'$r_{r+1}$' for r in range(nr-1)] + ["init"])
                    ax.set_title('Reward generation beliefs')

            fig.suptitle('Initial reward beliefs')

            '''Plot Environment reward generation matrix'''
            fig,axes = plt.subplots(1,na,figsize=(3*na,3))
            
            for ai, ax,title in zip(np.arange(na), axes, [f'Reward Regime {i}' for i in range(1,na+1)]):
                ax.xaxis.tick_top()
                sns.heatmap(reward_generation_matrix[:,:,ai],annot=True,ax=ax, cbar=False, cmap="viridis")
                ax.set_xticklabels([ f'L{i}' for i in range(1,na+1)] + ['init'],minor=False)
                ax.set_yticklabels([f'$r_{r+1}$' for r in range(0,nr-1)] + ['init'])
                ax.set_title(title)

        ##### Setup Classes
        env = MultiArmedBandit(state_transition_matrix,
                            reward_generation_matrix, 
                            TAU=TAU,
                            T=T,
                            n_bandits = nb,
                            training_protocol=training_protocol,
                            observation_generation_matrix=observation_generation_matrix,
                            context_observation_generation_matrix = np.eye(na),
                            no=no)


        agent = HDP_correct(lambda_H = lambda_H,
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
                    K = nc,
                    template_context_contingencies = template_context_contingencies,
                    context_observation_counts=context_observation_counts,
                    use_context_obs=use_context_obs,
                    gamma_init = gamma_init,
                    cap=cap,
                    use_template = use_template)



        world = World(agent, env, training_protocol=training_protocol)
        world.simulate_experiment()




        #### Find correct labels for inferred contexts
        best_fit = 0
        best_label = []

        for perm in range(1000):
            
            n_context = agent.K if agent.K >= na else na
            l = np.random.permutation(n_context)        
            labels = l[agent.context]
            fit = (agent.context == l[training_protocol]).sum()/TAU
            
            if fit > best_fit:
                best_fit = fit
                best_label = l
        
        
        
        a = agent.prior_rewards[-1,:nr-1,:,:agent.K]
        entropy = (-a*np.log(a)).sum(axis=0)
        learned_clear_winner = np.all(entropy < 0.6)            
        # print(f"best label assignment: {best_label}")

        Q_rew = agent.prior_rewards[-1,:,:,:agent.K]  + 1e-10      # inferred reward distribution given state and context
        plots = [Q_rew[:,:,k] for k in range(agent.K)]
        titles = [f"Context {k+1}" for k in range(agent.K)]
        # titles[0] = f"l: {best_label}, alpha: {round(alpha,6)}, gamma: {round(gamma,6)}", f"kappa: {round(kappa,6)}"

        ### plots inferred reward distributions for each context
        if plot_rewards:
            plots = [plot[:nr-1,:nb] / plot[:nr-1,:nb].sum(axis=0)[None,:]  for plot in plots]
            plot_rewards_heatmap(data=plots, title=titles, dpi=dpi,rewards=True)

        if plot_transition_matrix:
            plot_transition_matrix_heatmap(agent.transition_matrix[:agent.K+1,:agent.K+1].round(2),dpi=dpi)

        if plot_context_obs:
            plot_transition_matrix_heatmap(agent.prior_context_observation_counts[-1,:,:agent.K+1],dpi=dpi,vmax=None)

        ### CONTEXT PLOT
        
        data = agent
        post_context = data.posterior_context[:,:,:].copy()
        post_policies = np.nan_to_num(data.posterior_policies[:,:,:,:data.K])
        prior_policies = np.nan_to_num(data.prior_policies[:-1,:,:data.K])
        like_policies = np.nan_to_num(data.likelihood_policies[:,:,:,:data.K])
        actions = data.actions
        
        K = np.cumsum(agent.opened_new_context)+nc 
        novel_context = data.posterior_context[np.arange(TAU),:,K[:-1]]
        post_context[np.arange(TAU),:,K[:-1]] = 0
        post_context /= post_context.sum(axis=-1)[:,:,None]

        # all_labels = np.arange(max_context)
        # other_labels = [label for label in all_labels if label not in best_label]
        # post_context = post_context[:,:, list(best_label) + other_labels]

        new_context = (data.opened_new_context == True).nonzero()
            
        if plot_context:
            fig, ax = plt.subplots(1, figsize=(3,3), dpi=dpi)
            ax.set_ylim((0,1.05))
            plt.grid(axis="x", alpha=0.7)
            # plt.grid(axis="y", alpha=0.7)
            # plt.grid(axis="y")
            ax.xaxis.set_major_locator(MultipleLocator(switch[ind]))  # Set tick spacing on x-axis to 1
            # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
            # plt.hlines(0.5,xmin=0,xmax=switch*2, color = 'k', alpha=0.2)
            # ax.scatter(np.arange(TAU), y_val_action, marker="o", color="r", s=30)
            # ax.scatter(np.arange(TAU), y_val_unexp_event, marker="x", color="k")
            
            for c in range(data.K):
                ax.plot(post_context[:,1,c],label=f"Context {c+1}", linewidth=1)
            
            ax.plot(novel_context[:,1], 'gray', label=f"Novel\ncontext")
            
            for i, ind in enumerate(new_context):
                if i == len(new_context)-1:
                    ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5, label="Context\nopened")    
                else:
                    ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5)
            
            ax.set_title(f"{alpha},{gamma},{kappa}", fontsize=14, y = 1.05)
            ax.legend(
                bbox_to_anchor=[1.05,1.05],
                framealpha=1,
                labelspacing = 1,
                fontsize=14
                # loc='lower center',
                # bbox_to_anchor=(0.5, -0.8),
                # ncol=data.K+2,
                # framealpha=0,
                # fontsize=20
                )
            
            # ax.legend(bbox_to_anchor=[2,-0.22], framealpha=1, fontsize=14, ncols = agent.K+2)

            ax.set_xlabel("trial", fontsize=20)
            ax.tick_params(axis="x",labelsize=18, rotation = 40)
            ax.tick_params(axis="y",labelsize=18)

            plt.show()

            # fig, ax = plt.subplots(1, figsize=(3,3), dpi=dpi)
            # ax.set_ylim((0,1.05))
            # plt.grid(axis="x", alpha=0.7)
            # ax.xaxis.set_major_locator(MultipleLocator(switch[0]))
            
            # for c in range(data.K):
            #     ax.plot(data.q_c[:,-1,c],label=f"Context {c+1}", linewidth=1)
            
            # ax.plot(data.q_c[np.arange(TAU),-1,K[:-1]], 'gray', label=f"Novel\ncontext")
            
            # for i, ind in enumerate(new_context):
            #     if i == len(new_context)-1:
            #         ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5, label="Context\nopened")    
            #     else:
            #         ax.vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5)
            
            # ax.set_title(f"{gamma}; Posterior Context; correct in {(100*best_fit).round()}% of trials", fontsize=14, y = 1.05)
            # ax.legend(
            #     bbox_to_anchor=[1.05,1.05],
            #     framealpha=1,
            #     labelspacing = 1,
            #     fontsize=14
            #     )
            
            # ax.set_xlabel("trial", fontsize=20)
            # ax.tick_params(axis="x",labelsize=18, rotation = 40)
            # ax.tick_params(axis="y",labelsize=18)

            # plt.show()
            

            # fig, ax = plt.subplots(1, data.K, figsize=((data.K)*4,3), dpi=dpi)
            # plt.grid(axis="x", alpha=0.7)
            # plt.subplots_adjust(wspace = 1.4)
            # for m in range(data.K):
            #     ax[m].set_ylim((0,1.05))
            #     ax[m].xaxis.set_major_locator(MultipleLocator(switch[0]))
                
            #     for c in range(data.K):
            #         ax[m].plot(data.q_z[:,-1,m,c],label=f"Context {c+1}", linewidth=1)
                
            #     ax[m].plot(data.q_z[np.arange(TAU),-1,m,K[:-1]], 'gray', label=f"Novel\ncontext")
                
            #     for i, ind in enumerate(new_context):
            #         if i == len(new_context)-1:
            #             ax[m].vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5, label="Context\nopened")    
            #         else:
            #             ax[m].vlines(ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5)
                
            #     ax[m].set_title(rf"$q(z_{{{m}}})$", fontsize=14, y = 1.05)
                
            #     ax[m].set_xlabel("trial", fontsize=20)
            #     ax[m].tick_params(axis="x",labelsize=18, rotation = 40)
            #     ax[m].tick_params(axis="y",labelsize=18)

            # ax[0].legend(
            #     bbox_to_anchor=[2.1,1.05],
            #     framealpha=1,
            #     labelspacing = 1,
            #     fontsize=14
            #     # loc='lower center',
            #     # bbox_to_anchor=(0.5, -0.8),
            #     # ncol=data.K+2,
            #     # framealpha=0,
            #     # fontsize=20
            #     )
            # plt.show()
        
        
        if plot_messages:
            #### messages plot?
            counts = agent.global_prior_counts.copy()
            q_z = np.array([agent.construct_G_0(counts[trial]) for trial  in range(TAU)])
            obs_messages = np.nan_to_num(agent.context_likelihood)


            # result = q_z*obs_messages
            # result[result == 0] = -1000
            # result = np.argmax((result),axis=1)


            q_z[q_z == 0 ] = -1000
            z = np.argmax(q_z,axis=1)
            obs_messages[obs_messages==0] = -1000
            obs = np.argmax(obs_messages,axis=1)
            q_z[q_z == -1000 ] = None
            obs_messages[obs_messages == -1000] = 0


            K = agent.K+1 if agent.K % 2 == 1 else agent.K
            nrows = 2
            fig, axes = plt.subplots(nr, K // nrows, dpi=dpi+200)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.7)
            K = agent.K
            for ai, ax in enumerate(axes.flatten()):
                ax.grid()
                ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
                ax.plot(np.arange(TAU-1), obs_messages[1:,ai], linewidth=0.5)
                ax.plot(np.arange(TAU-1), q_z[1:,:K],'--', label = f"q_z {ai}", linewidth = 0.5)
                ax.set_ylabel(r"$\ln F(c)$",color="blue")
                ax2 = ax.twinx()
                color = 'orange'
                ax2.set_ylabel(f'$q_z \ln F(c)$', color=color)
                ax2.plot(np.arange(TAU-1), (q_z*obs_messages)[1:,ai], color="tab:orange",linewidth=0.5,alpha=0.6)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim([-1,0])
                ax.set_ylim([-4,1.5])
                
            plt.legend(bbox_to_anchor=[1,0.8])

            
        if plot_choice:
            df = pd.DataFrame({"trial": np.arange(TAU), "action": agent.actions[:,0], "context":training_protocol})
            plot_conditional_action_probs(df,dpi=dpi)

        print(f"rep: {best_fit.round(3)}")
        learned_correct.append(best_fit)

#%%
