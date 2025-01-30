#%%
import numpy as np
from scipy.special import digamma, softmax
import matplotlib.pyplot as plt
from environment import SpeakerDiscretizationEnvironment #data, W, component_params_true, transition_matrix_true, plot_heatmap # data - the words spoken, the speaker and the time point; W -  how many observations are passed at a time
import warnings
from itertools import product
import pandas as pd
from agent import HDP_speaker_discretization
from misc import plot_heatmap


#%% Create environment and data
env = SpeakerDiscretizationEnvironment()
env.plot_heatmap(data=env.transition_matrix_true, title="state transition matrix")
env.plot_speakers()

np.random.seed(6)
data = env.create_data(N=150, W=5)
env.plot_data(data)



#%% Run Simulations
# gammas = np.arange(0.001, 0.1,0.005)
# alphas = np.arange(0.001, 0.15,0.005)
# kappas = np.arange(0,0.01,0.001)

# gammas = np.arange(0.001, 0.15,0.005)
# alphas = np.arange(0.001, 0.15,0.005)
# kappas = np.arange(0,0.02,0.002)


######THESE WORK WELL WHEN same z is used for inference!
# gammas = np.arange(0.001, 0.15,0.005)
# alphas = np.arange(0.001, 0.014,0.001)
# kappas = np.arange(0,0.02,0.002)

######these work well when differnt z is used for inference!
gammas = np.arange(0.001, 0.1,0.001)
alphas = np.arange(0.004, 0.006, 0.0001)
kappas = np.arange(0.02,0.035,0.005)

# log for simulation data
simulation_data = np.zeros([int(gammas.size*alphas.size*kappas.size),5])
simulation_data.fill(np.nan)

i = 0
distance_best_fit = 10000

print(f"total number of simulations: {gammas.size*alphas.size*kappas.size}")
for gamma, alpha, kappa in product(gammas,alphas,kappas):
# for gamma, alpha, kappa in ([[0.121, 0.003, 0.01]]):

    # run inference
    agent = HDP_speaker_discretization()
    agent.initialize_HDP(TAU=data.shape[0],alpha=alpha, gamma=gamma, kappa=kappa, obs_batch_size=env.W)

    for tau in range(int(data[:,0].size / 5)): #]):
        agent.update_beliefs(data[tau*env.W:(tau+1)*env.W,0],tau)

    # if inferred the right number of contexts
    if agent.K == 3:
        Q_rew = agent.generative_model_obs[:,:agent.K]        # inferred speaker distributions over words
        P_rew = env.component_params_true                         # true speaker distributions over words

        labels = np.zeros(3,dtype=int)                  # which learned distribution corresponds to which true distribution 
        true_divergence = np.zeros(3)                   # distance between inferred and true distribution once labels allocated
        divergences = np.zeros((3,3))                   # distance between inferred distribution and all three possible true distributions

        # transition matrix with last row removed
        tm = (agent.transition_matrix[:agent.K,:agent.K]/agent.transition_matrix[:agent.K,:agent.K].sum(axis=0)[None,:])

        # label allocation
        for distribution in range(3):
            
            for candidate in range(3):
                
                # DKL[q||p]
                divergences[distribution, candidate] = (Q_rew[:,candidate]*np.log(Q_rew[:,candidate]/P_rew[:,distribution])).sum()
            
            # the label given is the one with the smallest divergance
            labels[distribution] = np.argmin(divergences[distribution])
            true_divergence[distribution] = divergences[distribution, np.argmin(divergences[distribution])]
            
        if np.unique(labels).size == 3:   # if distinct best candidate exists for each distribution

            total_distance = true_divergence.sum()/3 # calculate total fit quality

            # calculate how divergent inferred transition matrix is
            Q_rew = Q_rew[:,labels]
            Q_tm = tm[:,labels]
            Q_tm = Q_tm[labels,:]
            P_tm = env.transition_matrix_true

            total_distance_tm = (Q_tm*np.log(Q_tm/P_tm)).sum(axis=0).mean()

            title = f"{total_distance.round(3)}_{i}_" 
            plt.figure()
            plt.plot(Q_rew)
            plt.ylim(0,0.6)
            plt.savefig(title+"_0.png")
            plt.close()
            plot_heatmap(Q_tm.round(2), file_title = title + "_2", title=f"trans matrix - gamma: {round(gamma,3)}, alpha: {round(alpha,3)}, kappa: {round(kappa,3)}")
            simulation_data[i] = np.array([total_distance, gamma,alpha,kappa, total_distance_tm])
    else:
        print(f"{gamma},{alpha}, {kappa},{agent.K}")

    if i%100==0:
        print(i)

    i += 1
    

#%% Plot parameters and fit quality

df = pd.DataFrame(data = simulation_data, columns = ["distribution_distance","gamma","alpha","kappa","tm_distance"])
df = df.dropna()
# df = df[df["distribution_distance"]<0.5]
# df.drop(columns=["i","matrix_distance"])


# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Scatter plot with color mapping for performance
scatter = ax.scatter(df['alpha'], df['gamma'], df['kappa'], c=df['distribution_distance'], cmap='viridis', s=40)
# Add labels
ax.set_xlabel('alpha')
ax.set_ylabel('gamma')
ax.set_zlabel('kappa')
# Add a colorbar
cbar = fig.colorbar(scatter)
cbar.set_label('Average DKL[Q_reward||P_reward]')
plt.show()

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Scatter plot with color mapping for performance
scatter = ax.scatter(df['alpha'], df['gamma'], df['kappa'], c=df['tm_distance'], cmap='viridis', s=40)
# Add labels
ax.set_xlabel('alpha')
ax.set_ylabel('gamma')
ax.set_zlabel('kappa')
# Add a colorbar
cbar = fig.colorbar(scatter)
cbar.set_label('Average DKL[Q_tm||P_tm]')
plt.show()


print(df[df["distribution_distance"] == df["distribution_distance"].min()])
print(df[df["tm_distance"] == df["tm_distance"].min()])


