#%%
import numpy as np
np.set_printoptions(suppress=True)
# %matplotlib widget
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pandas as pd
import seaborn as sns
import scipy.special as scp
from itertools import product
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
# from utils import *
# from misc import *
from environment import MultiArmedBandit
from agent import HDP, HDP_IMM, HDP_correct
from world import World


np.random.seed(2)


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
        g = sns.heatmap(data=im, annot=True, annot_kws={"size":14}, cmap="viridis", cbar=False, fmt=fmt, ax=ax,vmin=vmin, vmax=vmax)
        # g.set_xlabel("states")
        # g.set_ylabel("rewards")

        if title is not None:
            ax.set_title(title[ai])
    
        ax.set_axis_off()

    fig.suptitle("Learned reward contingencies", fontsize=14, y=1.06, fontweight="bold" )
    # 
  
    if save:
        plt.savefig(file_title + ".png",dpi=300)
        plt.close()

    return fig, axes


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


def plot_conditional_action_probs(TAU, agent, training_protocol, context_col='context', action_col='action', base_palette='tab10',dpi=100):
   
    df = pd.DataFrame({"trial": np.arange(TAU), context_col: agent.actions[:,0], action_col:training_protocol})
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


def plot_context_posterior(post_context, novel_context, new_context, switch,inset=False):
    fig, ax = plt.subplots(1, figsize=(3,3), dpi=dpi)
    ax.set_ylim((0,1.05))
    plt.grid(axis="x", alpha=0.7)
    ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
    # plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
    # plt.hlines(0.5,xmin=0,xmax=switch*2, color = 'k', alpha=0.2)
    # ax.scatter(np.arange(TAU), y_val_action, marker="o", color="r", s=30)
    # ax.scatter(np.arange(TAU), y_val_unexp_event, marker="x", color="k")

    for c in range(data.K):
        # print(post_context[:,1,c])
        ax.plot(post_context[:,1,c],label=f"Context {c+1}", linewidth=1)

    ax.plot(novel_context[:,1], 'gray', label=f"Novel\ncontext")


    for ii, cont_ind in enumerate(new_context):
        if ii == len(new_context)-1:
            ax.vlines(cont_ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', linewidth=2, label="Context\nopened")    
        else:
            ax.vlines(cont_ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', linewidth=2, alpha=0.5)

    # ax.set_title(f"{gamma}; Posterior Context; correct in {(100*best_fit).round()}% of trials", fontsize=14, y = 1.05)
    # ax.legend(
    #     loc='lower center',
    #     bbox_to_anchor=(0.5, -0.8),
    #     ncol=data.K+2,
    #     framealpha=0,
    #     fontsize=20
    # )
    ax.legend(
        bbox_to_anchor=(1.06, 1),
        fontsize=14,
        labelspacing=1.3,
        alignment="right"
        
    )
    ax.set_ylabel("Posterior Context", fontsize=20)
    ax.set_xlabel("trial", fontsize=20)
    ax.tick_params(axis="x",labelsize=18, rotation = 40)
    ax.tick_params(axis="y",labelsize=18)

    if inset:
        # axins = inset_axes(ax, width="40%", height="40%", loc="upper right")  # inset size/position
        axins = inset_axes(
                            ax,
                            width="30%", height="70%",  # size relative to parent
                            bbox_to_anchor=(1.2,0.5, 1, 1),  # (x0, y0, width, height)
                            bbox_transform=ax.transAxes,
                            loc="lower left"
                            )
        for c in range(data.K):
            axins.plot(post_context[:,1,c], linewidth=1)

        axins.plot(novel_context[:,1], 'gray')
        axins.vlines(new_context,ymin=0,ymax=1.05, color = 'k', linestyle='--', alpha=0.5)
        axins.grid(axis="x", alpha=0.7)
        # focus region around trial 200
        zoom_width = 20   # number of trials around 200
        # axins.set_xlim(180, 220)
        axins.set_xlim(80,120)
        axins.set_ylim(0, 0.75)
        axins.vlines(200, ymin=0, ymax=0.75, color="gray", alpha=0.5)

        # axins.set_xticks([])
        # axins.set_yticks([])

        # optional: draw a box linking inset to main plot
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3")
    plt.show()
    
    # fig.savefig("fig3_posterior_trace"+".svg", dpi=300, bbox_inches='tight')


def plot_context_messages(agent,switch,TAU):
#### messages plot?
    counts = agent.global_prior_counts.copy()
    q_z = np.array([agent.digamma_approximation(counts[trial]) for trial  in range(TAU)])
    obs_messages = np.nan_to_num(agent.context_likelihood)


    result = q_z*obs_messages
    result[result == 0] = -1000
    result = np.argmax((result),axis=1)

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
#%%
# Task setup parameters
# single agent plots
plot_rewards = False
plot_transition_matrix = False
plot_context = True
plot_context_obs = False
plot_choice = False
plot_messages = False
plot_bad_fits = False

# paper plots and averaged agent plots
plot_avg_context_posterior_all = False         #  shows inferred posterior in all environments
plot_avg_context_posterior_template = False    #
plot_avg_entropy_accuracy_given_h = False
plot_avg_context_accuracy = False
plot_habit_benefit = False
plot_template_benefit = False
use_context_obs = False
debug = False
dpi = 100

repeats = 2

approx_pred_pol = True  # refers to whether digamma is used or not
approx_pred_rew = True

# example 2 armed bandit graph
# params_dict = {
#     "number_of_bandits" : np.array([3]), 
#     "switch" : np.array([100]),
#     "gammas" : np.array([900]),       # global prior context opening tendency
#     "alphas" : np.array([25]),        # local  prior context opening tendency
#     "kappas" : np.array([250]),       # self-transition bias
#     "hs"     : np.array([10000]),     # np.floor(np.exp(np.arange(1,9.5,0.25))), # 
#     "rho_global" : np.array([1]),     # global prior counts forgetting rate
#     "rho_local" : np.array([1]),      # local prior counts forgetting rate 
#     "use_template" : np.array([False]),
# }

# habit simulations
# params_dict = {
#     "number_of_bandits" : np.array([3]), 
#     "switch" : np.array([130]),
#     "gammas" : np.array([900]),       # global prior context opening tendency
#     "alphas" : np.array([25]),        # local  prior context opening tendency
#     "kappas" : np.array([250]),       # self-transition bias
#     "hs"     : np.array([10000]),#np.floor(np.exp(np.arange(1,9.5,0.25))), # np.array([10000])
#     "rho_global" : np.array([1]),     # global prior counts forgetting rate
#     "rho_local" : np.array([1]),      # local prior counts forgetting rate 
#     "use_template" : np.array([False]),
# }


# example 4 armed bandit params
params_dict = {
    "number_of_bandits" : np.array([4]), 
    "switch" : np.array([300]),
    "gammas" : np.array([850]),       # global prior context opening tendency
    "alphas" : np.array([30]),        # local  prior context opening tendency
    "kappas" : np.array([250]),       # self-transition bias
    "hs"     : np.array([10000]),  # np.floor(np.exp(np.arange(1,9.5,0.25))), # 
    "rho_global" : np.array([1]),     # global prior counts forgetting rate
    "rho_local" : np.array([1]),      # local prior counts forgetting rate 
    "use_template" : np.array([False,True]),
}


max_context = 7


sim_name = "test2.csv   "#"df_habit_accuracy_certainty.csv"
gamma_init = 1000
cap = 100000


sim_params = product(*params_dict.values())
reps = 1 # how many times to run simulation with same params


n_sims = 1
for key, val in params_dict.items():
    n_sims *= val.size
n_sims *=reps                        #alphas.size*kappas.size*gammas.size*hs.size*rho_global.size*rho_local.size*number_of_bandits.size*reps

print(f"-----------------------------------")
print(f"{n_sims} simulations to run")
i = -1

###### Run simulations
learned_correct = []

dfs = []
dfs_small = []

for na, switch, gamma, alpha, kappa, h, rho_l, rho_g, use_template in sim_params:  
    for rep in range(reps):
        nb = na
        ns = nb+1
        no = ns
        nco = na
        nr = 3
        nc = 0
        nt = na  # number of template contexts!
        T = 2
        npi = na**(T-1)

        # if rep == 1:
        #     debug = True
        # else:
        #     debug = False

        # ind = np.arange(params_dict["number_of_bandits"].size)[params_dict["number_of_bandits"] == nb][0]
        training_protocol = np.tile(np.arange(nb).repeat(switch),repeats)
        training_protocol = np.concatenate([training_protocol, np.tile(np.arange(nb).repeat(100),3)])
        
        # if h != 10000 and use_template:
        #     print("skipping h=20 with template")
        #     break
        if rep % 25 == 0:
            print(na, switch, gamma, alpha, kappa, h, use_template)
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
        
        lambda_H = np.ones([nr,ns])*5
        lambda_H[0,:-1] = 1
        lambda_H[-1,-1] = 100
        counts_prior_rewards = np.zeros([nr,nb+1,nc])#np.stack([lambda_H for i in range(nc)],axis=-1)
        
        bias = 1
        for c in range(nc):
            init_counts = np.ones([nr,nb+1])
            init_counts[1,c] = bias
            init_counts[0,np.arange(na+1) != c] = bias
            init_counts[:,-1] = [1,1,100]
            counts_prior_rewards[:,:,c] = init_counts
            counts_prior_rewards[:,:,c] += np.random.uniform(low=0, high=1, size=(nr,ns))
            
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
        utility = np.array([0.005, 0.99, 0.005]) # np.array([1/nr]*3) #


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
                            context_observation_generation_matrix = context_obs_generation_matrix,
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
                    K = nc,
                    template_context_contingencies = template_context_contingencies,
                    context_observation_counts=context_observation_counts,
                    use_context_obs=use_context_obs,
                    gamma_init = gamma_init,
                    cap=cap,
                    use_template = use_template,
                    # dec_temp = 2
                    )



        world = World(agent, env, training_protocol=training_protocol)
        world.simulate_experiment()

        #### Find correct labels for inferred contexts
        best_fit = 0
        best_label = []

        for permutation in range(1000):
            0
            n_context = agent.K if agent.K >= na else na               # how many contexts to find label for
            l = np.random.permutation(n_context)                       # pick random context labels
            # fit = (agent.context == l[training_protocol]).sum()/TAU    # relabel training protocol with random labels and see how many match inferred context 
            fit = (agent.context == l[training_protocol])[nb*switch:].sum()/(TAU//2)
            if fit > best_fit:
                best_fit = fit
                best_label = l


        # check if there is a clear winner for each bandit in each context
        a = agent.prior_rewards[-1,:nr-1,:,:agent.K]
        entropy = (-a*np.log(a)).sum(axis=0)
        learned_clear_winner = np.all(entropy < 0.6)

        learned_correct.append(best_fit)


        if plot_rewards:
            Q_rew = agent.prior_rewards[-1,:,:,:agent.K]  + 1e-10      # inferred reward distribution given state and context
            plots = [Q_rew[:,:,k] for k in range(agent.K)]
            titles = [f"Context {k+1}" for k in range(agent.K)]
            plots = [plot[:nr-1,:nb] / plot[:nr-1,:nb].sum(axis=0)[None,:]  for plot in plots]
            plot_rewards_heatmap(data=plots, title=titles, dpi=dpi,rewards=True)


        if plot_transition_matrix:
            plot_transition_matrix_heatmap(agent.transition_matrix[:agent.K+1,:agent.K+1].round(2),dpi=dpi)


        if plot_context_obs:
            plot_transition_matrix_heatmap(agent.prior_context_observation_counts[-1,:,:agent.K+1],dpi=dpi,vmax=None)

        ### CONTEXT PLOT
        
        # extract agent data
        data = agent
        post_context = data.posterior_context[:,:,:].copy()
        post_policies = np.nan_to_num(data.posterior_policies[:,:,:,:data.K])
        prior_policies = np.nan_to_num(data.prior_policies[:-1,:,:data.K])
        like_policies = np.nan_to_num(data.likelihood_policies[:,:,:,:data.K])
        actions = data.actions
        
        # separate novel context posterior and renormalize probabilities
        K = np.cumsum(agent.opened_new_context)+nc                      # how many contexts opened at each trial
        novel_context = data.posterior_context[np.arange(TAU),:,K[:-1]] # novel context posterior
        post_context[np.arange(TAU),:,K[:-1]] = 0                       # remove novel context from posterior matrix
        post_context /= post_context.sum(axis=-1)[:,:,None]             # renormalize without novel context
        
        # reorder contexts in terms of best fitting labels
        all_labels = np.arange(max_context)
        other_labels = [label for label in all_labels if label not in best_label]
        post_context = post_context[:,:, list(best_label) + other_labels]

        new_context = (data.opened_new_context == True).nonzero()
        # print(new_context)
        # print(new_context[0][best_label])


        if plot_context:
            plot_context_posterior(post_context, novel_context, new_context, switch,inset=False)

        if plot_messages:
            plot_context_messages(agent, switch, TAU)

        if plot_choice:
            plot_conditional_action_probs(TAU, agent, training_protocol, dpi=dpi)

        if plot_bad_fits and best_fit < 0.9:
            print(f"rep: {rep}")
            plot_context_posterior(post_context, novel_context, new_context, switch,inset=True)
            Q_rew = agent.prior_rewards[-1,:,:,:agent.K]  + 1e-10      # inferred reward distribution given state and context
            plots = [Q_rew[:,:,k] for k in range(agent.K)]
            titles = [f"Context {k+1}" for k in range(agent.K)]
    
            plots = [plot[:nr-1,:nb] / plot[:nr-1,:nb].sum(axis=0)[None,:]  for plot in plots]
            plot_rewards_heatmap(data=plots, title=titles, dpi=dpi,rewards=True)

            plot_conditional_action_probs(TAU, agent, training_protocol, dpi=dpi)

        
        ### create dataframe
        df = pd.DataFrame()
        post_context = post_context[:,1,:]
        for k in range(max_context):
            df[str(k)] = post_context[:,k]
        
        df[str(max_context)] = novel_context[:,1]
        df["nb"] = na
        df["agent"] = i
        df["rep"] = rep
        df["h"] = h
        df["gamma"] = gamma
        df["alpha"] = alpha
        df["kappa"] = kappa
        df["switch"] = switch
        repeated = np.ones(training_protocol.size)
        repeated[0:nb*switch] = 0
        df["repeated"] = repeated #np.array([0]*nb + [1]*nb*(repeats-1)).repeat(switch)
        df["phase"] = training_protocol
        df["block"] = np.concatenate([np.arange(nb*repeats).repeat(switch), np.arange(nb*repeats, nb*repeats + nb*3).repeat(100)])
        df["trial"] = np.arange(TAU)
        df["K"] = agent.K
        uniform = (post_context>0)*np.ones([TAU, max_context])/K[:-1][:,None]
        df["real_entropy"] = np.nan_to_num(-post_context*np.log(post_context)).sum(axis=1)
        df["entropy"] = np.nan_to_num(post_context*np.log(post_context/uniform)).sum(axis=1)
        df["choice"]  = agent.actions[:,0]
        df["reward"] =  agent.rewards[:,1]
        df["template"] = use_template + 0
        df["fit"] = learned_correct[-1]
        df["trial_n"] = df.groupby("block").cumcount()
        df["learned_clear_winner"] = learned_clear_winner

        dfs.append(df)


print(f"\n\n total: {(np.array(learned_correct) >= 0.9).sum()/n_sims}")
df_big = pd.concat(dfs).reset_index()
df_big.to_csv(sim_name, index=False)


#%% Plot effects of template and habituation
nb = 4
switch = 300
hs = [40,1000]
name = "new_table_data"#f"new_4_bandits_templates_and_h40"
ncols=nb+1
max_context = 7
fig_name = f"fig4_cont_{nb}" #name

plot_avg_context_posterior_all = True
#%%
if plot_avg_context_posterior_all:#

    #### conext posterior plot
    
    alphas = ['solid','dotted']
    cols =[["grey","#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], ["k"]*5]
    fig, axes = plt.subplots(1, ncols, dpi=300, figsize=(ncols*3, 3.3), sharey=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8)
    df_big = pd.read_csv(name+".csv")

    for hi, h in enumerate(hs):
        post_cols = set([str(el) for el in np.arange(max_context+1)])
        id_vars = set(df_big.columns).difference(post_cols)
        # ["index","trial","h","agent","phase","entropy","K", "nb"]
        print(df_big.shape)
        # df_big = df_big.query("fit >= 0.9")
        print(df_big.shape)
        df = pd.melt(df_big, id_vars=id_vars, var_name="context", value_name="post_context")

        axes[0].set_ylabel("Posterior Context", fontsize=22, labelpad=10)

        n_trials = df_big.query("agent == 1").repeated.to_numpy().size
        repeated = df_big.query("agent == 1").repeated.to_numpy()
        # Plot regular contexts in columns 1-4
        for i in range(0, nb+1):
            if i==0:
                print(f'h=={h} & context=="{max_context}" & nb=={nb}')
                sns.lineplot(
                    ax=axes[0],
                    data=df.query(f'h=={h} & context=="{max_context}" & nb=={nb}'),
                    x="trial", y="post_context", color=cols[hi][0], errorbar="se")#, linestyle=alphas[hi]
                # )
            else:
                print(f'h=={h} & context=="{i-1}" & nb=={nb}')  
                
                sns.lineplot(
                    ax=axes[i],
                    data=df.query(f'h=={h} & context=="{i-1}" & nb=={nb}'),
                    x="trial", y="post_context", color=cols[hi][i], errorbar="se")#, linestyle=alphas[hi]
                # )

                # display(df.query(f'h=={h} & context=="{i-1}" & nb=={nb}'))
            axes[i].set_xlabel(r"trial $\tau$", fontsize=22)
            axes[i].grid(axis="x", which="both", alpha=0.7)
            axes[i].set_ylim([-0.05,1.05])
            axes[i].tick_params(axis="x", labelrotation=35, labelsize=20)
            axes[i].tick_params(axis="y", labelsize=22)
            # axes[i].xaxis.set_minor_locator(MultipleLocator(switch[j]))
            axes[i].xaxis.set_major_locator(MultipleLocator(switch*2))   # x-labels every switch[j]*2
            axes[i].xaxis.set_minor_locator(MultipleLocator(switch))     # grid lines every switch[j]
        # Hide unused axes


    # Add context titles above the top row, colored by seaborn palette and larger font
    for col_idx, title in enumerate(context_titles):
        if col_idx == 0:
            axes[col_idx].set_title(title, fontsize=25, pad=20, color="grey",fontweight='bold')
        else:
            axes[col_idx].set_title(title, fontsize=25, pad=20, color=cols[0][col_idx],fontweight='bold')

#%%    ### mean accuracy plot
 
    fig = plt.figure(figsize=(4.5,4),dpi=300)
    plt.grid()
    df_big["optimal"] = df_big["choice"] == df_big["phase"]
    # df = df_big.groupby(by=["h","agent"])["optimal"].mean().reset_index()
    # g = sns.boxplot(data=df, x="h", y="optimal", palette="Greys_r")
    df = df_big.groupby(by=["h","agent","repeated"])["optimal"].mean().reset_index()
    g = sns.boxplot(data=df.query("repeated==1"), x="h",y="optimal",palette="Greys_r")

    # g = sns.boxplot(data=df.query("repeated=1"), x="h",y="optimal",hue="repeated",palette="Greys_r")
    g.set_ylim([0,1])
    g.set_ylabel("Mean Accuracy",fontsize=16)
    g.set_xticklabels(["Moderate\nautomatisation","No\nautomatisation"], fontsize=16)
    g.set_yticklabels(g.get_yticklabels(), fontsize=14)

    handles, labels = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=["New Context", "Familiar Context"], title="")
    g.set_xlabel("")
    fig.savefig("supp_fig_new.svg")
#%%    #### context detection benefit plot

    cut_off = 0.2
    df = df_big.query(f"fit>{cut_off}")
    # df = df_big.copy().query(f"h==10000 & fit>{cut_off}")
    print(df.template.unique())
    print(df.h.unique())
    print(df.query("template==0")["agent"].nunique())
    print(df.query("template==1")["agent"].nunique())

    masks = [(df[str(context)] >= 0.75) & (df["phase"] == context) for context in df["phase"].unique()]

    subset = df[masks[0] | masks[1] | masks[2] | masks[3]]

    keys = ["h","template","agent", "repeated", "phase","block"]

    all_idx = df.groupby(keys).size().index  # every group, even if empty after filtering

    df = (
        subset.groupby(keys)["trial_n"].min()  # first matching trial within each present group
        .reindex(all_idx)
        .reset_index(name="first_trial"))                      # add missing groups as NaN)

    df["first_trial"] = df["first_trial"].fillna(switch)
    df = df.query("~(phase == 0 & repeated == 0)")
    df["phase"] += 1
    fig, ax = plt.subplots(1,2, figsize=(8.5*1.3,3.2*1.1),dpi=300)
    plt.subplots_adjust(wspace=0.7)
    # fig, ax = plt.subplots(1,2, figsize=(10,4),dpi=300)

    titles = ["New context", "Familiar context"]
    # letters = ["A", "B"]
    for i in range(2):
        sns.barplot(ax=ax[i],data=df.query(f"repeated == {i}"), x="phase", y="first_trial", hue="h", palette="gray", edgecolor="black")
        ax[i].grid()
        # ax[i].set_ylabel(r"First trial where $p > 0.75$", fontsize=16)
        ax[i].set_ylabel(fr"First trial where $q(c=c_{{\text{{true}}}}) > 0.75$", fontsize=16, labelpad=10)
        ax[i].tick_params(labelsize=14)
        ax[i].set_ylim([0,200])
        ax[i].set_title(titles[i], fontsize=16,pad=20)
        # ax[i].set_xticklabels([i for i in range(1,5)])
        ax[i].set_xlabel("Context",fontsize=16)
        handles, _ = ax[i].get_legend_handles_labels()
        # labels = [f"h = {h_val}" for h_val in hs]
        labels = ["Moderate\nautomatisation","No\nautomatisation"]
        ax[i].legend().set_title(None)
        ax[i].legend(handles=handles, labels=labels,fontsize=14, loc="upper left")

#%% print table numbers
df_big = pd.read_csv("new_table_data.csv")

df_big["optimal"] = df_big["choice"] == df_big["phase"]
df_acc = df_big.groupby(["h","template","rep","repeated"])["optimal"].mean().to_frame().reset_index()
df_acc = df_acc.groupby(["h","template","repeated"]).agg(["mean", "sem"])["optimal"]
df_acc.columns = ["accuracy_mean", "accuracy_sem"]

##### select only posterior probability of currently active context
df_context = df_big[["h","template","phase","rep","trial","0","1","2","3"]]
df_context = df_context.melt(id_vars=["h","template","phase","rep","trial"], var_name="context", value_name="context_probability")
df_context["context"] = df_context["context"].astype(int)
df_context.loc[df_context["phase"] != df_context["context"], "context_probability"] = np.nan

#### calculate mean active context probabiity per agent and then, mean and SEM of the sample
df_context = df_context.groupby(["h","template","rep"])["context_probability"].mean()
df_context = df_context.groupby(["h","template"]).agg(["mean", "sem"])
df_context.columns = ["context_mean", "context_sem"]

df = pd.concat([df_context, df_acc.loc[:,:,1]],axis=1).round(3)
df.to_csv("summary_table.csv")

#%% Plot average context accuracy
plot_avg_context_accuracy = True
# if plot_avg_context_accuracy:
#     nb = 4
#     df = pd.read_csv(f"{nb}_bandits_good_h70.csv") #df_big.copy()
#     df["learned_correct"] = df["fit"] >= 0.9
#     df["chose_correct"] = df["choice"] == df["phase"]
#     df['trial_n'] = df.groupby(['agent','h','phase']).cumcount() + 1
#     df['cum_correct'] = df.groupby(['agent','h','phase'])['chose_correct'].cumsum()
#     df['cum_accuracy'] = df['cum_correct'] / df['trial_n']

#     fig, axes = plt.subplots(int(np.ceil(nb/2)),2,dpi=300)
#     axes = axes.flatten()
#     for ai, ax in enumerate(axes):
#         sns.lineplot(ax=ax, data=df.query(f"phase == {ai}"), x="trial_n", y="cum_accuracy",\
#                     hue="learned_correct", errorbar="se")
#         ax.set_ylabel("Cummulative accuracy")
#         ax.set_xlabel("Trial")
#         ax.set_title(f"Context {ai+1}")
#         ax.set_ylim([0,1])
#         ax.grid()
#         # ax.legend(title=r"$\alpha_{init}$")
#     plt.tight_layout()

#%%
if plot_avg_context_accuracy:
    nb = 3
    df = pd.read_csv(f"dec_temp3_3.csv") #df_big.copy()
    df["learned_correct"] = df["fit"] >= 0.9
    # df = df.query("learned_correct == True")
    df["chose_correct"] = df["choice"] == df["phase"]
    df['trial_n'] = df.groupby(['agent','h','block']).cumcount() + 1
    df['cum_correct'] = df.groupby(['agent','h','repeated','phase'])['chose_correct'].cumsum()
    df['cum_accuracy'] = df['cum_correct'] / df['trial_n']

    fig, axes = plt.subplots(int(np.ceil(nb/2)),2,dpi=300)
    axes = axes.flatten()
    for ai, ax in enumerate(axes):
        sns.lineplot(ax=ax, data=df.query(f"phase == {ai}"), x="trial_n", y="cum_accuracy",\
                    hue="repeated", errorbar="se")
        ax.set_ylabel("Cummulative accuracy")
        ax.set_xlabel("Trial")
        ax.set_title(f"Context {ai+1}")
        ax.set_ylim([0,1])
        ax.grid()
        # ax.legend(title=r"$\alpha_{init}$")
    plt.tight_layout()



#%% Plot individual context posterior
nb = 4
switch = 300
h = 10000
name = "test"#f"new_{nb}_bandits_good"
# fig_name = f"fig4_cont_{nb}"#name
# panel_labels = ["C"]
df_big = pd.read_csv(name+".csv")
plot_avg_context_posterior_all = True
if plot_avg_context_posterior_all:#
    post_cols = set([str(el) for el in np.arange(max_context+1)])
    id_vars = set(df_big.columns).difference(post_cols)
    # ["index","trial","h","agent","phase","entropy","K", "nb"]
    total = df_big.shape[0]
    # df_big = df_big.query("fit >= 0.9")
    # cleaned =  df_big.shape[0]
    # print(cleaned/total)
    
    df = pd.melt(df_big, id_vars=id_vars, var_name="context", value_name="post_context")
    display(df.head())
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    # cols = ["tab10:blue", "tab10:orange", "tab10:green", "tab10:red"]

    ncols = 5
    context_titles = ["novel context", "context 1", "context 2", "context 3", "context 4"]

    fig, axes = plt.subplots(1, ncols, dpi=300, figsize=(ncols*3, 3.3), sharey=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8)


    axes[0].set_ylabel("Posterior context", fontsize=22, labelpad=10)

    n_trials = df_big.query("agent == 1").repeated.to_numpy().size
    repeated = df_big.query("agent == 1").repeated.to_numpy()
    # Plot regular contexts in columns 1-4
    for i in range(0, nb+1):
        if i==0:
            sns.lineplot(
                ax=axes[0],
                data=df.query(f'h=={h} & context=="{max_context}" & nb=={nb}'),
                x="trial", y="post_context", color="grey", errorbar="se",linewidth=3
            )
        else:
            sns.lineplot(
                ax=axes[i],
                data=df.query(f'h=={h} & context=="{i-1}" & nb=={nb}'),
                x="trial", y="post_context", color=cols[i-1], errorbar="se",linewidth=3
            )
    
            # axes[i].axvspan(switch*(i-1), switch*(i), color="gray", alpha=0.05, zorder=0)            
            # axes[i].axvspan(switch*(i-1+nb), switch*(i+nb), color="gray", alpha=0.09, zorder=0)            

        for ii in range(1,nb+1):
            axes[i].axvspan(switch*(ii-1), switch*(ii), color=cols[ii-1], alpha=0.1, zorder=0)            
            axes[i].axvspan(switch*(ii-1+nb), switch*(ii+nb), color=cols[ii-1], alpha=0.1, zorder=0)


            # display(df.query(f'h=={h} & context=="{i-1}" & nb=={nb}'))
        # axes[i].grid(axis="x", which="both", alpha=0.7)
        axes[i].set_xlabel(r"trial $\tau$", fontsize=22)
        axes[i].set_ylim([-0.05,1.05])
        axes[i].tick_params(axis="x", labelrotation=35, labelsize=20)
        axes[i].tick_params(axis="y", labelsize=22)
        # axes[i].xaxis.set_minor_locator(MultipleLocator(switch[j]))
        axes[i].xaxis.set_major_locator(MultipleLocator(switch*2))   # x-labels every switch[j]*2
        axes[i].xaxis.set_minor_locator(MultipleLocator(switch))     # grid lines every switch[j]
    # Hide unused axes
    for ax in axes[nb+1:]:
        ax.axis('off')

    # Add context titles above the top row, colored by seaborn palette and larger font
    for col_idx, title in enumerate(context_titles):
        ax = axes[col_idx]
        if col_idx == 0:
            ax.set_title(title, fontsize=25, pad=20, color="grey",fontweight='bold')
        else:
            ax.set_title(title, fontsize=25, pad=20, color=cols[col_idx-1],fontweight='bold')


# fig.savefig(fig_name+".svg", dpi=300, bbox_inches='tight')


#%% Plot benefit of encountering context again
from statannotations.Annotator import Annotator

plot_reoccuring_context_benefit = True
plot_contexts_together = True
if plot_reoccuring_context_benefit:
    switch = 100
    nb = 3
    for cut_point in [0.75]:
        df = pd.read_csv(f"{nb}_bandits_good.csv")


        masks = [(df[str(context)] >= cut_point) & (df["phase"] == context) for context in df["phase"].unique()]
        subset = df[masks[0] | masks[1] | masks[2]]

        keys = ["h","template","agent", "repeated", "phase"]

        all_idx = df.groupby(keys).size().index  # every group, even if empty after filtering

        df = (
            subset.groupby(keys)["trial_n"].min()  # first matching trial within each present group
            .reindex(all_idx)
            .reset_index(name="first_trial"))                      # add missing groups as NaN)

        df["first_trial"] = df["first_trial"].fillna(switch)
        df = df.query("~(phase == 0)") #& first_trial < 100")
        df["phase"] += 1

        # fig, ax = plt.subplots(1,1, figsize=(4,3.2), dpi=100)
        # plt.subplots_adjust(wspace=0.7,top=0.95)

        fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=300)
        # plt.subplots_adjust()

        ax.grid(zorder=-10)

        if plot_contexts_together:
            sns.barplot(ax=ax, data=df, x="repeated", y="first_trial", palette="gray", edgecolor="black")# hue="repeated"
            order = sorted(df["repeated"].unique())
            annot = Annotator(ax, [(0,1)], data=df, x="repeated", y="first_trial", order=order)
            annot.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
            annot.apply_test()
            ax, test_results = annot.annotate()
            ax.set_xticks([0,1],["New\nContext","Familiar\nContext"])
            ax.tick_params(axis="x", labelsize=14)
            ax.set_xlabel("")
            ax.set_ylim([0,60])
                        
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(bbox_to_anchor=(1, 0.7),handles=handles, labels=["New\nContext","Familiar\nContext"],fontsize=14)# ax.legend(["New Context", "Familiar Context"], loc="center right")

        else:
            sns.barplot(ax=ax, data=df, x="phase", y="first_trial", hue="repeated", palette="gray", edgecolor="black")
            ax.set_xlabel("Context",fontsize=14)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(bbox_to_anchor=(1, 0.7),handles=handles, labels=["New\nContext","Familiar\nContext"],fontsize=14)# ax.legend(["New Context", "Familiar Context"], loc="center right")

            # Prepare the box pairs for statannotations
            unique_phases = sorted(df["phase"].unique())
            box_pairs = [((phase, 0), (phase, 1)) for phase in unique_phases]
            # Create the annotator
            annotator = Annotator(ax, pairs=box_pairs, data=df,x="phase", y="first_trial", hue="repeated")
            annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", comparisons_correction=None, verbose=0)
            
            _,results = annotator.apply_and_annotate()
            
            for res in results:
                print(res.data)
        
        # ax.set_ylabel("First trial where\n"+fr"$q(c=c_{{\text{{true}}}}) > {cut_point}$", fontsize=16, labelpad=10)
        ax.set_ylabel(fr"First trial where $q(c=c_{{\text{{true}}}}) > {cut_point}$", fontsize=14, labelpad=10)
        ax.tick_params(axis="both", labelsize=12)
        
        fig.savefig(f"fig5.svg", dpi=300)#, bbox_inches='tight')

        # fig, axes = plt.subplots(1,2)
        # for ai, ax in enumerate(axes):
        #     sns.histplot(data=df.query(f"phase == {ai+2}"), x="first_trial", hue="repeated",ax=ax)


#%% Plot context posterior with and without automatization for n=4

plot_avg_context_posterior_all = True
nb = 4
switch = 300
h = 40

df_names = ["new_4_bandits_good.csv", f"new_4_bandits_good_h{h}.csv"]
# cols = ["tab10:blue", "tab10:orange", "tab10:green", "tab10:red"]

ncols = 5
context_titles = ["Novel Context", "Context 1", "Context 2", "Context 3", "Context 4"]

# fig, axes = plt.subplots(1, ncols, dpi=300, figsize=(ncols*3, 3.3), sharey=True)
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.8)

dfs = []
for name in df_names:
    dfs.append(pd.read_csv(name))

alphas = ['solid','dotted']
cols =[["k"]*5,["grey","#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]]
fig, axes = plt.subplots(1, ncols, dpi=300, figsize=(ncols*3, 3.3), sharey=True)
plt.tight_layout()
plt.subplots_adjust(hspace=0.8)

# for name in df_names:
    # df_big = pd.read_csv(name)

if plot_avg_context_posterior_all:#
    for dfi, df_big in enumerate(dfs):
        h = df_big["h"].unique()[0]
        print(h)
        post_cols = set([str(el) for el in np.arange(max_context+1)])
        id_vars = set(df_big.columns).difference(post_cols)
        # ["index","trial","h","agent","phase","entropy","K", "nb"]
        print(df_big.shape)
        # df_big = df_big.query("fit >= 0.9")
        print(df_big.shape)
        df = pd.melt(df_big, id_vars=id_vars, var_name="context", value_name="post_context")

        axes[0].set_ylabel("Posterior Context", fontsize=22, labelpad=10)

        n_trials = df_big.query("agent == 1").repeated.to_numpy().size
        repeated = df_big.query("agent == 1").repeated.to_numpy()
        # Plot regular contexts in columns 1-4
        for i in range(0, nb+1):
            if i==0:
                print(f'h=={h} & context=="{max_context}" & nb=={nb}')
                sns.lineplot(
                    ax=axes[0],
                    data=df.query(f'h=={h} & context=="{max_context}" & nb=={nb}'),
                    x="trial", y="post_context", color=cols[dfi][0], errorbar="se")#, linestyle=alphas[dfi]
                # )
            else:
                print(f'h=={h} & context=="{i-1}" & nb=={nb}')  
                
                sns.lineplot(
                    ax=axes[i],
                    data=df.query(f'h=={h} & context=="{i-1}" & nb=={nb}'),
                    x="trial", y="post_context", color=cols[dfi][i], errorbar="se")#, linestyle=alphas[dfi]
                # )

                # display(df.query(f'h=={h} & context=="{i-1}" & nb=={nb}'))
            axes[i].set_xlabel(r"trial $\tau$", fontsize=22)
            axes[i].grid(axis="x", which="both", alpha=0.7)
            axes[i].set_ylim([-0.05,1.05])
            axes[i].tick_params(axis="x", labelrotation=35, labelsize=20)
            axes[i].tick_params(axis="y", labelsize=22)
            # axes[i].xaxis.set_minor_locator(MultipleLocator(switch[j]))
            axes[i].xaxis.set_major_locator(MultipleLocator(switch*2))   # x-labels every switch[j]*2
            axes[i].xaxis.set_minor_locator(MultipleLocator(switch))     # grid lines every switch[j]
        # Hide unused axes


# Add context titles above the top row, colored by seaborn palette and larger font
for col_idx, title in enumerate(context_titles):
    if col_idx == 0:
        axes[col_idx].set_title(title, fontsize=25, pad=20, color="grey",fontweight='bold')
    else:
        axes[col_idx].set_title(title, fontsize=25, pad=20, color=cols[1][col_idx],fontweight='bold')

# title = f"M=4"
# # Row title below panel label, left side
# axes[0].annotate(
#             title, xy=(-0.2, 0.5), xycoords='axes fraction', fontsize=25,
#             color='black', ha='right', va='center', rotation=0, annotation_clip=False,
#             xytext=(-60, 0), textcoords='offset points'
#             )


# # Panel label at top left corner of each row
# axes[0].annotate( "A", xy=(-0.8, 1.5), xycoords='axes fraction', fontsize=28, 
#                  fontweight='bold', ha='left', va='top', annotation_clip=False, 
#                  xytext=(-60, 0), textcoords='offset points'
# )


fig.savefig(f"habit_h{h}-1000.svg", dpi=300, bbox_inches='tight')



#%% Plot effect of habitual tendency on relative context entropy and choice accuracy for different H
plot_avg_entropy_accuracy_given_h = True
df_big = pd.read_csv("new_habit_accuracy_sims.csv")
#%%
if  plot_avg_entropy_accuracy_given_h:
    # plt.rcParams['font.family'] = 'DejaVu Sans'
    # plt.rcParams['text.usetex'] = False  # crucial

    # # Change only math font
    # plt.rcParams['mathtext.fontset'] = 'cm'   # or 'dejavusans', 'cm', 'stixsans', etc.

    if False:
        # get context with highest posterior probability in a trial
        df_big["inferred_context"] = df_big[["0","1","2","3","4","5"]].idxmax(axis=1)
        df_big["inferred_context"] = pd.to_numeric(df_big["inferred_context"], errors='coerce').astype('Int64')
        # check if correct context was inferred
        df_big["inferred_correctly"] = df_big["inferred_context"] == df_big["phase"]
        # calculate half of trials, after which training regime repeats
        half_of_trials = df_big.switch.unique()[0]*(df_big.phase.max()+1)
        
        # df_big["groupwise_fit"] = df_big.groupby(["h","agent","repeated"])["inferred_correctly"].transform('sum')/half_of_trials
        # df_big["learned_correct_structure"] = df_big["groupwise_fit"] > 0.9
        # sns.lineplot(data=df_big, x="h",y="learned_correct_structure", hue="repeated")

        # calculate groupwise accuracy based on whether context was encountered for the first time or not
        df  = (df_big.groupby(["h","agent","repeated"])["inferred_correctly"].sum()/half_of_trials).reset_index()
        df["learned_correctly"] = df["inferred_correctly"] > 0.9
        df = (df.groupby(["h","repeated"])["learned_correctly"].sum()/100).reset_index()
        sns.lineplot(data=df, x="h",y="learned_correctly", hue="repeated")

    number_of_bandits = 4
    hs = np.floor(np.exp(np.arange(1,9.5,0.25))) 
    df = df_big.copy().query(f"repeated == 1")

    grouped = df.groupby(by=["h","agent"])[["entropy","reward"]].mean().reset_index()
    grouped["h"] = np.log(1/grouped["h"])
    # display(grouped)

    fig, ax = plt.subplots(1,2, figsize=(8.5*1.3,3*1.2),dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.78)

    sns.lineplot(ax=ax[0], data=grouped, x="h", y="entropy", errorbar="se",marker="o",markersize=5)#,capsize=0.25,markersize=2)

    ax[0].set_xlabel(r"Automatization tendency $\ln h$",fontsize=22, labelpad=10)
    ax[0].set_ylabel(r"$D_{KL}\left[q(c) | p_{\text{unif}}(c)\right]$", fontsize=24, labelpad=10)
    ax[0].set_ylim([0.52,1.08])
    ax[0].axvspan(-3.8, -2.3, color="gray", alpha=0.2, zorder=0)            
    ax[0].xaxis.set_label_coords(0.4,-0.15)

    ax[0].tick_params(labelsize=16)

    sns.lineplot(ax=ax[1], data=grouped, x="h", y="reward", errorbar="se", marker="o", markersize=5)#,capsize=0.25,markersize=2)
    ax[1].set_xlabel(r"Automatization tendency $\ln h$",fontsize=22, labelpad=10,x=0.43)
    ax[1].set_ylabel(f"Mean accuracy", fontsize=22, labelpad=15)
    ax[1].set_ylim([0,0.8])
    ax[1].tick_params(labelsize=14)
    ax[1].axhline(y=0.25,  color="gray", alpha=0.7, zorder=0, label="chance level")
    ax[1].axvspan(-3.8, -2.3, color="gray", alpha=0.2, zorder=0)
    ax[1].xaxis.set_label_coords(0.4,-0.15)
         
    # ax[1].legend()

    panel_labels = ["B", "C"]
    # text_xy =  [(np.log(1/90), 0.93), (np.log(1/90), 0.73)]
    # point_xy = [(np.log(1/70), 0.87), (np.log(1/70), 0.65)]

    text_xy =  [(np.log(1/40), 1.06), (np.log(1/40), 0.78)]
    point_xy = [(np.log(1/22), 1.02), (np.log(1/21), 0.71)]

    for i, axis in enumerate(ax):
        axis.grid(alpha=0.5)
        axis.annotate("", xytext=text_xy[i], xy=point_xy[i], arrowprops=dict(arrowstyle="->"))

        # axis.annotate(panel_labels[i], xy=(-0.4, 1.3), xycoords='axes fraction', fontsize=22, fontweight='bold', ha='left', va='top')

    axis.annotate( "Chance level", xy=(0.63, 0.4), xycoords='axes fraction', 
                  fontsize=14, ha='left', va='top')

    fig.savefig("habit_entropy_accuracy.svg",dpi=300,bbox_inches='tight')
#%% Plot template benefit

plot_template_benefit = True
switch = 300
if plot_template_benefit:
    for cut_off in [0.2]:#,0.9]:
        df = pd.read_csv("new_df_habit_template_benefit_11.csv").query(f"h==10000 & fit>{cut_off}")
        # df = df_big.copy().query(f"h==10000 & fit>{cut_off}")
        print(df.template.unique())
        print(df.h.unique())
        print(df.query("template==0")["agent"].nunique())
        print(df.query("template==1")["agent"].nunique())

        masks = [(df[str(context)] >= 0.75) & (df["phase"] == context) for context in df["phase"].unique()]

        subset = df[masks[0] | masks[1] | masks[2] | masks[3]]

        keys = ["h","template","agent", "repeated", "phase","block"]

        all_idx = df.groupby(keys).size().index  # every group, even if empty after filtering

        df = (
            subset.groupby(keys)["trial_n"].min()  # first matching trial within each present group
            .reindex(all_idx)
            .reset_index(name="first_trial"))                      # add missing groups as NaN)

        df["first_trial"] = df["first_trial"].fillna(switch)
        df = df.query("~(phase == 0 & repeated == 0)")
        df["phase"] += 1
        fig, ax = plt.subplots(1,2, figsize=(8.5*1.3,3.2*1.1),dpi=300)
        plt.subplots_adjust(wspace=0.7)
        # fig, ax = plt.subplots(1,2, figsize=(10,4),dpi=300)

        titles = ["New context", "Familiar context"]
        # letters = ["A", "B"]
        for i in range(2):
            sns.barplot(ax=ax[i],data=df.query(f"repeated == {i}"), x="phase", y="first_trial", hue="template", palette="gray", edgecolor="black")
            ax[i].grid()
            # ax[i].set_ylabel(r"First trial where $p > 0.75$", fontsize=16)
            ax[i].set_ylabel(fr"First trial where $q(c=c_{{\text{{true}}}}) > 0.75$", fontsize=16, labelpad=10)
            ax[i].tick_params(labelsize=14)
            ax[i].set_ylim([0,200])
            ax[i].set_title(titles[i], fontsize=16,pad=20)
            # ax[i].set_xticklabels([i for i in range(1,5)])
            ax[i].set_xlabel("Context",fontsize=16)
            handles, _ = ax[i].get_legend_handles_labels()
            labels = ["Without template","With template"]
            ax[i].legend().set_title(None)
            ax[i].legend(handles=handles, labels=labels,fontsize=14, loc="upper left")
            # ax[i].annotate(
            #     letters[i],
            #     xy=(-0.1, 1.3),
            #     xycoords='axes fraction',
            #     fontsize=25,
            #     color='black',
            #     ha='right',
            #     va='center',
            #     rotation=0,
            #     annotation_clip=False,
            #     xytext=(-60, 0),
            #     textcoords='offset points'
            # )

        # fig.savefig("fig8_temp_benefit.svg")

#%% OLD SBDM poster figure

plot_avg_context_posterior_all = True
nb = 4
switch = 100
h = 1000
max_context = 7
df_names = ["new_4_bandits_bad.csv", f"new_4_bandits_good_temp.csv"]
# cols = ["tab10:blue", "tab10:orange", "tab10:green", "tab10:red"]

ncols = 5
context_titles = ["Novel Context", "Context 1", "Context 2", "Context 3", "Context 4"]

# fig, axes = plt.subplots(1, ncols, dpi=300, figsize=(ncols*3, 3.3), sharey=True)
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.8)

dfs = []
for name in df_names:
    dfs.append(pd.read_csv(name))

alphas = ['solid','dotted']
cols =[["k"]*5,["grey","#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]]
fig, axes = plt.subplots(1, ncols, dpi=300, figsize=(ncols*3, 3.3), sharey=True)
plt.tight_layout()
plt.subplots_adjust(hspace=0.8)

# for name in df_names:
    # df_big = pd.read_csv(name)

if plot_avg_context_posterior_all:#
    for dfi, df_big in enumerate(dfs):
        h = df_big["h"].unique()[0]
        print(h)
        post_cols = set([str(el) for el in np.arange(max_context+1)])
        id_vars = set(df_big.columns).difference(post_cols)
        # ["index","trial","h","agent","phase","entropy","K", "nb"]
        print(df_big.shape)
        # df_big = df_big.query("fit >= 0.9")
        print(df_big.shape)
        df = pd.melt(df_big, id_vars=id_vars, var_name="context", value_name="post_context")

        axes[0].set_ylabel("Posterior Context", fontsize=22, labelpad=10)

        n_trials = df_big.query("agent == 1").repeated.to_numpy().size
        repeated = df_big.query("agent == 1").repeated.to_numpy()
        # Plot regular contexts in columns 1-4
        for i in range(0, nb+1):
            if i==0:
                print(f'h=={h} & context=="{max_context}" & nb=={nb}')
                sns.lineplot(
                    ax=axes[0],
                    data=df.query(f'h=={h} & context=="{max_context}" & nb=={nb}'),
                    x="trial", y="post_context", color=cols[dfi][0], errorbar="se")#, linestyle=alphas[dfi]
                # )
            else:
                print(f'h=={h} & context=="{i-1}" & nb=={nb}')  
                
                sns.lineplot(
                    ax=axes[i],
                    data=df.query(f'h=={h} & context=="{i-1}" & nb=={nb}'),
                    x="trial", y="post_context", color=cols[dfi][i], errorbar="se")#, linestyle=alphas[dfi]
                # )

                # display(df.query(f'h=={h} & context=="{i-1}" & nb=={nb}'))
            axes[i].set_xlabel(r"trial $\tau$", fontsize=22)
            axes[i].grid(axis="x", which="both", alpha=0.7)
            axes[i].set_ylim([-0.05,1.05])
            axes[i].tick_params(axis="x", labelrotation=35, labelsize=20)
            axes[i].tick_params(axis="y", labelsize=22)
            # axes[i].xaxis.set_minor_locator(MultipleLocator(switch[j]))
            axes[i].xaxis.set_major_locator(MultipleLocator(switch*2))   # x-labels every switch[j]*2
            axes[i].xaxis.set_minor_locator(MultipleLocator(switch))     # grid lines every switch[j]
        # Hide unused axes


# Add context titles above the top row, colored by seaborn palette and larger font
for col_idx, title in enumerate(context_titles):
    if col_idx == 0:
        axes[col_idx].set_title(title, fontsize=25, pad=20, color="grey",fontweight='bold')
    else:
        axes[col_idx].set_title(title, fontsize=25, pad=20, color=cols[1][col_idx],fontweight='bold')


