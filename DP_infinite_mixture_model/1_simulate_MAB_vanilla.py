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
from agent import Agent
from world import World
from perception import HierarchicalPerception


np.random.seed(11)
plt.rcParams['figure.dpi'] = 100


def plot_rewards_heatmap(data, file_title=str(0), title=None,vmin=0,vmax=1,save=False,dpi=300, rewards=False):
    
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
        g = sns.heatmap(data=im, annot=True, annot_kws={"size":16}, cmap="viridis", cbar=False, fmt='.2f', ax=ax,vmin=vmin, vmax=vmax)
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

na = 4
nb = na
ns = nb+1
no = ns
nr =  3
nc = na
T = 2
npi = na**(T-1)


plot_rewards = True
plot_context = True
plot_context_obs = True
plot_choice = True
plot_messages = False
debug = False
dpi = 100

switch = 100
repeats = 1
training_protocol = np.tile(np.arange(nb).repeat(switch),repeats)
# training_protocol = np.concatenate([training_protocol, np.array([nb-1]).repeat(switch)])

# plt.rcParams['axes.xaxis.major.locator'] = MultipleLocator(switch)
TAU = training_protocol.size

approx_pred_pol = True  # refers to whether digamma is used or not
approx_pred_rew = True

hs = np.array([1000])


reps = 1   # how many times to run simulation with same params
sim_params = product(hs)

for h in sim_params:
	for rep in range(reps):

		'''           define policies            '''
		# policies = list(product(list(np.arange(na))*(T-1)))
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
                  perception,
				  debug=debug
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
		
		if plot_context:
			fig, ax = plt.subplots(1, figsize=(3,3), dpi=dpi)
			ax.set_ylim((0,1.05))
			plt.grid(axis="x", alpha=0.7)
			# plt.grid(axis="y", alpha=0.7)
			# plt.grid(axis="y")
			ax.xaxis.set_major_locator(MultipleLocator(switch))  # Set tick spacing on x-axis to 1
			# plt.vlines(switch,ymin=0,ymax=1, color = 'k', linestyle='--', alpha=0.5)
			# plt.hlines(0.5,xmin=0,xmax=switch*2, color = 'k', alpha=0.2)
			# ax.scatter(np.arange(TAU), y_val_action, marker="o", color="r", s=30)
			# ax.scatter(np.arange(TAU), y_val_unexp_event, marker="x", color="k")		      
			for c in range(nc):
				ax.plot(post_context[:,1,c],label=f"Context {c+1}", linewidth=2)		         
			ax.legend(bbox_to_anchor=[1.05,1.05], framealpha=1, labelspacing = 1, fontsize=14)
			# ax.legend(bbox_to_anchor=[2,-0.22], framealpha=1, fontsize=14, ncols = agent.K+2      
			ax.set_xlabel("trial", fontsize=14)
			ax.tick_params(labelsize=14)#, rotation = 45)		  
		
		if plot_choice:
			df = pd.DataFrame({"trial": np.arange(TAU), "action": actions[:,0], "context":training_protocol})
			plot_conditional_action_probs(df,dpi=dpi)		