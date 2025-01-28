#%%

import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from misc import plot_heatmap

##%% 


"""
Define true underlying HDP-HMM
We can imagine we are modeling a conversation between three people.
Each say W=5 words at a time. Every time a speaker finishes speaking,
a new speaker says another 5 words. The probability of who the next
speaker depends on who the last speaker was. 

Each speaker has the same vocabulary of 10 words, which they use with
different frequencies. So we havea HDP over the distributions over
words for each speaker? If successfull, the algorithm should infer three speakers 
with a high probability?
"""
M = 10      # number of observation categories
K_true = 3  # true number of mixture components


### define mixture components
component_params_true = np.zeros([M,K_true])

k = 8
weights = np.exp(np.linspace(0,1,10)*k).round(5)
weights_norm = weights/weights.sum()

weights  = np.array([.22,.15,.07, 0.04, 0.02])
weights = np.hstack((np.flip(weights),weights))

component_params_true[:,2] = weights_norm
component_params_true[:,1] = weights
component_params_true[:,0] = np.flip(weights_norm)

plt.plot(component_params_true,'-o')
print(component_params_true.sum(axis=0))

### define transition dynamics
transition_matrix_true = np.array([[.7,  .05 , .1                    ],
                                   [.15, .6  , .1],
                                   [.15, .35 , .8 ]])

# transition_matrix_true = np.array([[1/3,  1/3 , 1/3],
#                                    [1/3, 1/3  , 1/3],
#                                    [1/3, 1/3 , 1/3]])

plot_heatmap(transition_matrix_true);



"""Simulate HDP-HMM"""
np.random.seed(6)
N = 150
W = 5
data = np.zeros([N*W,2],dtype=int)   # first column for observed value, second column for what component
data[0,1] = 1            # the middle speaker says the first word
  
for n in range(N):
    for w in range(W):
        i = n*W+w
        speaker = data[i, 1]
        word = np.random.choice(np.arange(M), p=component_params_true[:,speaker])
        data[i,0] = word
    
        if w < W-1:
            next_speaker = speaker
        else: 
            next_speaker = np.random.choice(np.arange(K_true), p=transition_matrix_true[:, speaker])
        if i != N*W-1:
            data[i+1,1] = next_speaker

data = np.hstack((data, np.arange(data.shape[0])[:,None]))

plt.figure()
sns.scatterplot(data=pd.DataFrame(data,columns=["data","component","t"],),
             x = "t", y="data", hue="component", palette="viridis")

# plt.figure()
# sns.histplot(data[:,0], stat="probability")


# plt.figure()
# sns.histplot(data[:,1], stat="probability")