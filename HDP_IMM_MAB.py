#%%
import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp

from matplotlib.ticker import MultipleLocator
from itertools import product
from scipy.optimize import curve_fit
from statannotations.Annotator import Annotator


from environment import MultiArmedBandit
from agent import HDP_IMM
from world import World

np.set_printoptions(suppress=True)
np.random.seed(2)
dpi = 100

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


def plot_conditional_action_probs(TAU, agent, training_protocol, executed_action='context', true_context='action', base_palette='tab10',dpi=100):
   
    # prepare dataframe with trial number, chosen action and true context
    df = pd.DataFrame({"trial": np.arange(TAU), executed_action: agent.actions[:,0].astype(int), true_context: training_protocol})
    df["context"] += 1


    # count and normalize
    contexts = sorted(df[executed_action].unique())
    actions = sorted(df[true_context].unique())      
    counts = df.groupby([executed_action, true_context]).size().reset_index(name='count')
    counts['percent'] = counts.groupby(executed_action)['count'].transform(lambda x: x / x.sum())

    # generate color mapping
    if isinstance(base_palette, str):
        base_colors = sns.color_palette(base_palette, len(contexts))
    else:
        base_colors = base_palette
    context_color_map = dict(zip(contexts, base_colors))

    # function to generate RGBA with alpha varying by action index
    def get_alpha_color(base_color, idx, total):
        alpha = 1.175 -  0.7 * (idx + 1) / total  # Scale alpha from 0.4 to 1.0
        return (*base_color, alpha)

    # apply color with alpha per (context, possible action) pair
    counts['color'] = counts.apply(
        lambda row: get_alpha_color(
            context_color_map[row[executed_action]],
            actions.index(row[true_context]),
            len(actions)
        ),
        axis=1
    )

    # Plotting
    fig, ax = plt.subplots(1, figsize=(4,3),dpi=dpi)
    bar_width = 0.8 / len(actions)
    x = np.arange(len(contexts))

    for i, action in enumerate(actions):
        subset = counts[counts[true_context] == action]
        offsets = x + (i - len(actions)/2) * bar_width + bar_width/2
        ax.bar(
            offsets,
            subset['percent'],
            width=bar_width,
            color=subset['color'],
            edgecolor='black',
            linewidth=0.8,
            label=f'Action {action}'
        )

    # Grayscale legend for actions
    gray_shades = [str(0.1 + 0.7 * (i + 1) / len(actions)) for i in range(len(actions))]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=gray_shades[i])
        for i in range(len(actions))
    ]
    plt.legend(handles, [f"Action {int(a+1)}" for a in actions], bbox_to_anchor=[1,0.85], fontsize=8.5)

    plt.yticks(fontsize=12)
    plt.xticks(x, contexts, fontsize=12)
    plt.xlabel("Context",fontsize=12)
    plt.ylabel("Action execution distribution",fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_context_posterior(post_context, novel_context, new_context, switch):

    fig, ax = plt.subplots(1, figsize=(3,3), dpi=dpi)
    ax.set_ylim((0,1.05))
    plt.grid(axis="x", alpha=0.7)
    ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1


    for c in range(data.K):
        # print(post_context[:,1,c])
        ax.plot(post_context[:,1,c],label=f"Context {c+1}", linewidth=1)

    ax.plot(novel_context[:,1], 'gray', label=f"Novel\ncontext")


    for ii, cont_ind in enumerate(new_context):
        if ii == len(new_context)-1:
            ax.vlines(cont_ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', linewidth=2, label="Context\nopened")    
        else:
            ax.vlines(cont_ind,ymin=0,ymax=1.05, color = 'k', linestyle='--', linewidth=2, alpha=0.5)

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

    
    # fig.savefig("fig3_posterior_trace"+".svg", dpi=300, bbox_inches='tight')


"""Choose what simulation to run and figure to plot"""

# Single agent plots - can look at individual simulations

plot_rewards = False               # plots learned distribution over rewards for each agent instance
plot_transition_matrix = False     # plots learned transition matrix for each agent instance
plot_context = False               # plots inferred posterior over contexts for each agent instance
plot_choice = False                # plots action distribution in each context


# Plot given paper figure; only one of these should be true at a time when running script

plot_avg_context_posterior_M2 = False    # plot average posterior over contexts for M=2 bandit simulation (Fig.4A)
plot_avg_context_posterior_M3 = False    # plot average posterior over contexts for M=3 bandit simulation (Fig.4B and Fig.5)
plot_avg_context_posterior_M4 = False    # plot average posterior over contexts for M=4 bandit simulation. Can be used to plot (Fig.6) and (Fig.8C) by varying params_dict values accordingly.
plot_habit_benefit_posterior = False     # plot comparison of inferred posterior over contexts for automatising and non-automatising agents (Fig.7A)
plot_habit_vs_accuracy_context_certainty = False # plot context certainty and behavioural accuracy vs automatisation tendenct (Fig.7B) and (Fig.7C)
plot_template_benefit = True

# do not change these values, they are initialized here and changed within if else loop below

plot_reoccuring_context_benefit = False
plot_avg_context_posterior = False


if plot_avg_context_posterior_M2:
    
    plot_avg_context_posterior = True

    params_dict = {
        "number_of_bandits" : np.array([2]), 
        "switch" : np.array([100]),         # after how many trials context siwtches
        "gammas" : np.array([1000]),        # global prior context opening tendency
        "alphas" : np.array([20]),          # local  prior context opening tendency
        "kappas" : np.array([250]),         # self-transition bias
        "hs"     : np.array([10000]),       # automatisation tendency
        "use_template" : np.array([False]), # whether to use templates
    }
    
    reps = 20 # how many times to run simulation with same params
    sim_name = "df_MAB_M2.csv"   # name for csv file


elif plot_avg_context_posterior_M3:

    plot_avg_context_posterior = True
    plot_reoccuring_context_benefit = True
    
    params_dict = {
        "number_of_bandits" : np.array([3]), 
        "switch" : np.array([100]),         # after how many trials context siwtches
        "gammas" : np.array([900]),         # global prior context opening tendency
        "alphas" : np.array([25]),          # local  prior context opening tendency
        "kappas" : np.array([250]),         # self-transition bias
        "hs"     : np.array([10000]),       # automatisation tendency
        "use_template" : np.array([False]), # whether to use templates
    }
    
    reps = 20 # how many times to run simulation with same params
    sim_name = "df_MAB_M3.csv"   # name for csv file
    
    
elif plot_avg_context_posterior_M4:
    
    plot_avg_context_posterior = True

    params_dict = {
        "number_of_bandits" : np.array([4]), 
        "switch" : np.array([300]),         # after how many trials context siwtches
        "gammas" : np.array([850]),         # global prior context opening tendency
        "alphas" : np.array([30]),          # local  prior context opening tendency
        "kappas" : np.array([250]),         # self-transition bias
        "hs"     : np.array([10000]),       # automatisation tendency
        "use_template" : np.array([False]), # whether to use templates
    }

    reps = 20 # how many times to run simulation with same params
    sim_name = f"df_MAB_M4_switch{params_dict['switch'][0]}_h{params_dict['hs'][0]}.csv"   # name for csv file
    print(f"Using training regime with {params_dict['switch'][0]} trials per context.\nSet params_dict['switch'] to 100 for short training regime and 300 for long training regime.")

   
elif plot_habit_benefit_posterior:

    params_dict = {
        "number_of_bandits" : np.array([4]), 
        "switch" : np.array([300]),         # after how many trials context siwtches
        "gammas" : np.array([850]),         # global prior context opening tendency
        "alphas" : np.array([30]),          # local  prior context opening tendency
        "kappas" : np.array([250]),         # self-transition bias
        "hs"     : np.floor([40]),          # automatisation tendency
        "use_template" : np.array([False]), # whether to use templates
    }

    reps = 20 # how many times to run simulation with same params
    sim_name = f"df_MAB_M4_switch{params_dict['switch'][0]}_h{params_dict['hs'][0]}.csv"  
    # sim_name = f"df_MAB_M4_h{params_dict['hs'][0]}.csv"


elif plot_habit_vs_accuracy_context_certainty:

    params_dict = {
        "number_of_bandits" : np.array([4]), 
        "switch" : np.array([300]),         # after how many trials context siwtches
        "gammas" : np.array([850]),         # global prior context opening tendency
        "alphas" : np.array([30]),          # local  prior context opening tendency
        "kappas" : np.array([250]),         # self-transition bias
        "hs"     : np.floor(np.exp(np.arange(1,9.5,0.25))), # automatisation tendency
        "use_template" : np.array([False]), # whether to use templates
    }

    reps = 3 # how many times to run simulation with same params
    sim_name = f"df_MAB_M4_h_vs_accuracy_certainty.csv"
    
    
elif plot_template_benefit:

    params_dict = {
        "number_of_bandits" : np.array([4]), 
        "switch" : np.array([300]),         # after how many trials context siwtches
        "gammas" : np.array([850]),         # global prior context opening tendency
        "alphas" : np.array([30]),          # local  prior context opening tendency
        "kappas" : np.array([250]),         # self-transition bias
        "hs"     : np.floor([10000]),          # automatisation tendency
        "use_template" : np.array([True,False]), # whether to use templates
    }

    reps = 20 # how many times to run simulation with same params
    sim_name = f"df_MAB_M4_switch{params_dict['switch'][0]}_h{params_dict['hs'][0]}_template.csv"  
    # sim_name = f"df_MAB_M4_h{params_dict['hs'][0]}.csv"

""" Run simulations for figure """

max_context = 7                  # max possible contexts, beyond this number of opened contexts code will break
sim_params = product(*params_dict.values()) # all simulation value combinations

n_sims = 1
for key, val in params_dict.items():
    n_sims *= val.size
n_sims *=reps

print(f"-----------------------------------")
print(f"{n_sims} simulations to run")
i = -1

learned_correct = []  # array storing whether agent succesfully acquiered contextual structure in that simulatiom
dfs = []              # array holding dataframes from individual simulations

for na, switch, gamma, alpha, kappa, h, use_template in sim_params:  
    for rep in range(reps):
        
        i+=1
        gamma_init = 1000  # initial pseudocount when opening a new stickbreak
        cap = 100000       # global prior pseudocount cap
        nb = na            # number of bandits
        ns = nb+1          # number of states
        no = ns            # number of observations
        nco = na           # number of context observations
        nr = 3             # number of rewards
        nc = 0             # iniitial number of contexts
        nt = na            # number of templates
        T = 2              # episode length
        npi = na**(T-1)    # number of policies
        repeats = 2        # how many time thew training protocol repeats

        training_protocol = np.tile(np.arange(nb).repeat(switch),repeats)
        TAU = training_protocol.size  # number of trials
        

        if rep % 25 == 0:
            print(na, switch, gamma, alpha, kappa, h, use_template)



        ####### Setup simulation
        

        '''           define policies            '''
        policies = np.array(list(product( np.arange(na), repeat= T-1)))


        '''       define p(s_t|s_t-1, a_t-1)      '''

        prior_states = np.array([0]*nb + [1])
        state_transition_matrix = np.array([np.eye(nb+1)]*nb).transpose([1,2,0])
        state_transition_matrix[:,-1,:] = np.eye(nb+1)[:,:-1]


        '''          define p(o_t|s_t)            '''
        observation_generation_matrix = np.eye(ns)



        '''           define p(r|s,c)             '''
        
        lambda_H = np.ones([nr,ns])
        lambda_H[0,:-1] = 1
        lambda_H[-1,-1] = 100
        counts_prior_rewards = np.zeros([nr,nb+1,nc])
        
        bias = 1
        for c in range(nc):
            init_counts = np.ones([nr,nb+1])
            init_counts[1,c] = bias
            init_counts[0,np.arange(na+1) != c] = bias
            init_counts[:,-1] = [1,1,100]
            counts_prior_rewards[:,:,c] = init_counts
            counts_prior_rewards[:,:,c] += np.random.uniform(low=0, high=1, size=(nr,ns))
            
            
        '''          create templates            '''
        
        template_context_contingencies = np.ones([nr,nb+1,na])
        bias = 10
        for temp in range(0,nt):
            template_context_contingencies[1,temp,temp] = bias
            template_context_contingencies[0, np.arange(na+1) != temp, temp] = bias
            template_context_contingencies[:,-1,:] = np.array([1,1,100])[:,None]
        template_context_contingencies += np.random.uniform(low=0, high=1, size=(nr,ns,nt))


        prior_rewards = scp.digamma(counts_prior_rewards) - scp.digamma(counts_prior_rewards.sum(axis=0))
        prior_rewards = scp.softmax(prior_rewards, axis=0)




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


        '''          define p(d|c) - context observation generation matrix - not used in paper simulations           '''
        p = 0.9
        q = 1-p
        context_obs_generation_matrix = np.ones([nco, na])*(q/(na-1))
        context_obs_generation_matrix[np.arange(nco), np.arange(nco)] = p


        '''           define counts alpha in p(pi|theta,alpha)           '''
        counts_prior_policies = np.zeros([npi,nc+1]) + h

        prior_policies = scp.softmax(scp.digamma(counts_prior_policies) - scp.digamma(counts_prior_policies.sum(axis=0)))



        '''       define dummy utility RV p(R=1) '''
        utility = np.array([0.005, 0.99, 0.005]) # np.array([1/nr]*3) #


        '''   define context obs contingencies '''
        context_observation_counts = np.ones([nco,nc+1])


        ##### Initialize environment and agent classes
        
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
                    approx_pred_pol = True,
                    approx_pred_rew = True,
                    h = h,
                    debug=False,
                    rho_l= 1,
                    rho_g = 1,
                    max_context=max_context,
                    K = nc,
                    template_context_contingencies = template_context_contingencies,
                    context_observation_counts=context_observation_counts,
                    use_context_obs=False,
                    gamma_init = gamma_init,
                    cap=cap,
                    use_template = use_template)


        ### run simulations
        world = World(agent, env, training_protocol=training_protocol)
        world.simulate_experiment()

        #### Find correct labels for inferred contexts
        best_fit = 0
        best_label = []

        for permutation in range(1000):
            n_context = agent.K if agent.K >= na else na                 # how many contexts to find label for
            l = np.random.permutation(n_context)                         # pick random context labels
            fit = (agent.context == l[training_protocol])[nb*switch:].sum()/(TAU//2) # relabel training protocol with random labels and see how many match inferred context 
            if fit > best_fit:
                best_fit = fit
                best_label = l


        if plot_rewards:
            Q_rew = agent.prior_rewards[-1,:,:,:agent.K]  + 1e-10      # inferred reward distribution given state and context
            plots = [Q_rew[:,:,k] for k in range(agent.K)]
            titles = [f"Context {k+1}" for k in range(agent.K)]
            plots = [plot[:nr-1,:nb] / plot[:nr-1,:nb].sum(axis=0)[None,:]  for plot in plots]
            plot_rewards_heatmap(data=plots, title=titles, dpi=dpi,rewards=True)


        if plot_transition_matrix:
            plot_transition_matrix_heatmap(agent.transition_matrix[:agent.K+1,:agent.K+1].round(2),dpi=dpi)


        
        # extract agent data
        post_context = agent.posterior_context[:,:,:].copy()
        post_policies = np.nan_to_num(agent.posterior_policies[:,:,:,:agent.K])
        prior_policies = np.nan_to_num(agent.prior_policies[:-1,:,:agent.K])
        like_policies = np.nan_to_num(agent.likelihood_policies[:,:,:,:agent.K])
        actions = agent.actions
        
        # separate novel context posterior and renormalize probabilities
        K = np.cumsum(agent.opened_new_context)+nc                      # how many contexts opened at each trial
        novel_context = agent.posterior_context[np.arange(TAU),:,K[:-1]] # novel context posterior
        post_context[np.arange(TAU),:,K[:-1]] = 0                       # remove novel context from posterior matrix
        post_context /= post_context.sum(axis=-1)[:,:,None]             # renormalize without novel context
        
        # reorder contexts in terms of best fitting labels
        all_labels = np.arange(max_context)
        other_labels = [label for label in all_labels if label not in best_label]
        post_context = post_context[:,:, list(best_label) + other_labels]

        new_context = (agent.opened_new_context == True).nonzero()



        if plot_context:
            plot_context_posterior(post_context, novel_context, new_context, switch)

        if plot_choice:
            plot_conditional_action_probs(TAU, agent, training_protocol, dpi=dpi)
        
        ### create individial simulation dataframe
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
        df["repeated"] = np.array([0]*nb + [1]*nb*(repeats-1)).repeat(switch)
        df["phase"] = training_protocol
        df["block"] = np.arange(nb*repeats).repeat(switch)
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
        dfs.append(df)
 
df_big = pd.concat(dfs).reset_index()
df_big.to_csv(sim_name, index=False)


"""           Plot simulation results            """

if plot_avg_context_posterior:
    
    nb = params_dict["number_of_bandits"][0]
    switch = params_dict["switch"][0]
    h = params_dict["hs"][0]
    
    df_big = pd.read_csv(sim_name)

    post_cols = set([str(el) for el in np.arange(max_context+1)])
    id_vars = set(df_big.columns).difference(post_cols)
    total = df_big.shape[0]

    df = pd.melt(df_big, id_vars=id_vars, var_name="context", value_name="post_context")
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    ncols = 5
    context_titles = ["novel context", "context 1", "context 2", "context 3", "context 4"]

    fig, axes = plt.subplots(1, ncols, dpi=300, figsize=(ncols*3, 3.3), sharey=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8)

    axes[0].set_ylabel("Posterior context", fontsize=22, labelpad=10)

    n_trials = df_big.query("agent == 1").repeated.to_numpy().size
    repeated = df_big.query("agent == 1").repeated.to_numpy()

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
            

        for ii in range(1,nb+1):
            axes[i].axvspan(switch*(ii-1), switch*(ii), color=cols[ii-1], alpha=0.1, zorder=0)            
            axes[i].axvspan(switch*(ii-1+nb), switch*(ii+nb), color=cols[ii-1], alpha=0.1, zorder=0)


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
        elif col_idx < nb+1:
            ax.set_title(title, fontsize=25, pad=20, color=cols[col_idx-1],fontweight='bold')

 
if plot_reoccuring_context_benefit:
    
    switch = params_dict["switch"][0]
    nb = params_dict["number_of_bandits"][0]
    cut_point = 0.75
        
    df = pd.read_csv("simulation_df.csv")

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

    fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=300)

    ax.grid(zorder=-10)

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
                    
    ax.set_ylabel(fr"First trial where $q(c=c_{{\text{{true}}}}) > {cut_point}$", fontsize=14, labelpad=10)
    ax.tick_params(axis="both", labelsize=12)

  
if plot_habit_benefit_posterior:
    
    nb = params_dict["number_of_bandits"][0]
    switch = params_dict["switch"][0]

    df_names = ["df_MAB_M4_switch300_h10000.csv", f"df_MAB_M4_switch300_h{params_dict['hs'][0]}.csv"]
    # cols = ["tab10:blue", "tab10:orange", "tab10:green", "tab10:red"]

    ncols = 5
    context_titles = ["Novel Context", "Context 1", "Context 2", "Context 3", "Context 4"]


    dfs = []
    try:
        for name in df_names:
            dfs.append(pd.read_csv(name))

        alphas = ['solid','dotted']
        cols =[["k"]*5,["grey","#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]]
        fig, axes = plt.subplots(1, ncols, dpi=300, figsize=(ncols*3, 3.3), sharey=True)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.8)

        # for name in df_names:
            # df_big = pd.read_csv(name)


        for dfi, df_big in enumerate(dfs):
            h = df_big["h"].unique()[0]
            post_cols = set([str(el) for el in np.arange(max_context+1)])
            id_vars = set(df_big.columns).difference(post_cols)
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
                        x="trial", y="post_context", color=cols[dfi][0], errorbar="se")
                else:
                    print(f'h=={h} & context=="{i-1}" & nb=={nb}')  
                    
                    sns.lineplot(
                        ax=axes[i],
                        data=df.query(f'h=={h} & context=="{i-1}" & nb=={nb}'),
                        x="trial", y="post_context", color=cols[dfi][i], errorbar="se")

                axes[i].set_xlabel(r"trial $\tau$", fontsize=22)
                axes[i].grid(axis="x", which="both", alpha=0.7)
                axes[i].set_ylim([-0.05,1.05])
                axes[i].tick_params(axis="x", labelrotation=35, labelsize=20)
                axes[i].tick_params(axis="y", labelsize=22)
                axes[i].xaxis.set_major_locator(MultipleLocator(switch*2))   # x-labels every switch[j]*2
                axes[i].xaxis.set_minor_locator(MultipleLocator(switch))     # grid lines every switch[j]
            # Hide unused axes
    except Exception as e:
        print(e)
        print("\nA simulation file is missing. To plot automatisation benefit in terms of contextual inference and structure learning:\n\n"+
              "1. run code with plot_avg_context_posterior_M4 = True with long training regime of switch = 300\n"+
              "to simulate naive structure learnings with 300 trials per context.\n"
              "2. run code with plot_habit_benefit_posterior = True to simulate structure learners\n"+
              "which can automatise and plot figure.")


    # Add context titles above the top row, colored by seaborn palette and larger font
    for col_idx, title in enumerate(context_titles):
        if col_idx == 0:
            axes[col_idx].set_title(title, fontsize=25, pad=20, color="grey",fontweight='bold')
        else:
            axes[col_idx].set_title(title, fontsize=25, pad=20, color=cols[1][col_idx],fontweight='bold')


if  plot_habit_vs_accuracy_context_certainty:
    df_big = pd.read_csv(sim_name)
    number_of_bandits = params_dict["number_of_bandits"]
    hs = params_dict["hs"]
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
         

    text_xy =  [(np.log(1/40), 1.06), (np.log(1/40), 0.78)]
    point_xy = [(np.log(1/22), 1.02), (np.log(1/21), 0.71)]

    for i, axis in enumerate(ax):
        axis.grid(alpha=0.5)
        axis.annotate("", xytext=text_xy[i], xy=point_xy[i], arrowprops=dict(arrowstyle="->"))

    axis.annotate( "Chance level", xy=(0.63, 0.4), xycoords='axes fraction', 
                  fontsize=14, ha='left', va='top')

    
if plot_template_benefit:

    df = pd.read_csv(sim_name)

    masks = [(df[str(context)] >= 0.75) & (df["phase"] == context) for context in df["phase"].unique()]

    subset = df[masks[0] | masks[1] | masks[2] | masks[3]]

    keys = ["h","template","agent", "repeated", "phase","block"]

    all_idx = df.groupby(keys).size().index  # every group, even if empty after filtering

    df = (
        subset.groupby(keys)["trial_n"].min()  # first matching trial within each present group
        .reindex(all_idx)
        .reset_index(name="first_trial"))      # add missing groups as NaN)

    df["first_trial"] = df["first_trial"].fillna(switch)
    df = df.query("~(phase == 0 & repeated == 0)")
    df["phase"] += 1
    fig, ax = plt.subplots(1,2, figsize=(8.5*1.3,3.2*1.1),dpi=300)
    plt.subplots_adjust(wspace=0.7)

    titles = ["New context", "Familiar context"]

    for i in range(2):
        sns.barplot(ax=ax[i],data=df.query(f"repeated == {i}"), x="phase", y="first_trial", hue="template", palette="gray", edgecolor="black")
        ax[i].grid()
        ax[i].set_ylabel(fr"First trial where $q(c=c_{{\text{{true}}}}) > 0.75$", fontsize=16, labelpad=10)
        ax[i].tick_params(labelsize=14)
        ax[i].set_ylim([0,200])
        ax[i].set_title(titles[i], fontsize=16,pad=20)
        ax[i].set_xlabel("Context",fontsize=16)
        handles, _ = ax[i].get_legend_handles_labels()
        labels = ["Without template","With template"]
        ax[i].legend().set_title(None)
        ax[i].legend(handles=handles, labels=labels,fontsize=14, loc="upper left")

