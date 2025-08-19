
#%%

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import jsonpickle



def plot_heatmap(data, file_title=str(0), title=None,vmin=0,vmax=1,save=False,dpi=300, novel_ind=2):


    fig, axes = plt.subplots(1, len(data), figsize=(2.1*len(data),2))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    for ai, ax, im in zip(np.arange(len(data)), axes, data):
        sns.heatmap(data=im, annot=True, annot_kws={"size":16}, cmap="viridis", cbar=False, fmt='.2f', ax=ax,vmin=vmin, vmax=vmax)
        ax.set_ylabel("s",fontsize=14, labelpad=10)
        ax.invert_yaxis()
        if ai < novel_ind-1:
            ax.set_xlabel(fr"$p(r|s,\phi_{{{ai+1}}})$", fontsize=16)
        else:
            ax.set_xlabel(fr"$p(r|s,\phi^{{\mathrm{{novel}}}})$", fontsize=16)
            # fr"$p(r|s,\phi_{{{col_idx+1}}}^{{\mathrm{{true}}}})$"
        ax.set_yticklabels(np.arange(1,3))
    return fig, axes



contingencies = [np.array([[0.84,0.16], [0.16, 0.84]]), np.ones([2,2])*0.5]
titles = [r"$p(r|s,c,\phi_1)$", r"$p(r|s,c,\phi_2)$"]
fig, axes = plot_heatmap(contingencies,dpi=200, novel_ind=2)


contingencies = [np.array([[0.87,0.13], [0.13, 0.87]]), np.array([[0.12,0.88], [0.92, 0.08]]), np.ones([2,2])*0.5]
fig, axes = plot_heatmap(contingencies,dpi=200, novel_ind=3)



#%%
 
import numpy as np
np.set_printoptions(suppress=True)
# %matplotlib widget
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special as scp
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["text.usetex"] = False

number_bandits = np.array([2, 3, 4])
n_rows = len(number_bandits)
n_cols = 5

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols *3, n_rows * 3.5))
plt.tight_layout()
plt.subplots_adjust(wspace=0.6, hspace=0.8)
fig.set_dpi(300)

# Get seaborn default color palette
palette = sns.color_palette()

context_titles = ["context 1", "context 2", "context 3", "context 4", "training regime"]

# Create custom colormaps for each context column based on seaborn palette
custom_cmaps = []
for color in palette[:4]:
    # Create a colormap from white to the context color
    cmap = LinearSegmentedColormap.from_list(f'custom_cmap_{color}', [(1,1,1), color])
    custom_cmaps.append(cmap)

for row_idx, nb in enumerate(number_bandits):

    nr = 2   # number of rewards
    ns = nb  # number of states
    bandits = np.arange(nb)

    p = 0.9
    q = 1 - p

    reward_generation_matrix = np.ones([nr, ns, len(bandits)]) * q

    for context, b in enumerate(bandits):
        reward_generation_matrix[1, b, context] = p
        reward_generation_matrix[0, np.arange(nb) != b, context] = p

    # Plot heatmaps for each bandit
    for col_idx in range(4):
        ax = axes[row_idx, col_idx]
        if col_idx < nb:
            im = reward_generation_matrix[:, :, col_idx].T  # Transpose so x=reward, y=bandit
            cmap = custom_cmaps[col_idx]  # Use custom colormap for each context column
            sns.heatmap(im, annot=True, annot_kws={"size": 20}, cbar=False, fmt='.2f', ax=ax, vmin=0, vmax=1,cmap="viridis")# , cmap=cmap)
            ax.set_xlabel(fr"$p(r|s,\phi_{{{col_idx+1}}}^{{\mathrm{{true}}}})$", fontsize=22)
            
            ax.set_ylabel("s", fontsize=22, labelpad=10)
            ax.tick_params(axis='both', labelsize=16)
            ax.xaxis.set_label_position('bottom')
            ax.yaxis.set_label_position('left')
            ax.invert_yaxis()  # Invert y-axis to match bandit order
            # ax.set_xticks([])
            ax.set_yticklabels(np.arange(1, nb+1))            
        else:
            ax.axis('off')  # Leave empty

    # Placeholder plot in last column
    ax = axes[row_idx, 4]
    for b in range(nb):
        y = np.round(reward_generation_matrix[1,:,b].repeat(100))
        x = np.arange(1, nb*100+1)
        ax.plot(x , y, linewidth=3)  # Placeholder line
    ax.set_xlabel("trial", fontsize=20)
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_label_position('left')
    ax.spines[['right','top']].set_visible(False)
    ax.set_ylabel("Context Active", fontsize=22)
    ax.tick_params(axis='both', labelsize=16)

    # Show tickmarks and tick labels for last column
    # Remove these lines for last column:
    # ax.set_xticks([])
    ax.set_yticks([0,1])

# Add context titles above the top row, colored by seaborn palette and larger font
for col_idx, title in enumerate(context_titles):
    ax = axes[0, col_idx]
    if col_idx < 4:
        ax.set_title(title, fontsize=32, color=palette[col_idx], pad=60)
    else:
        ax.set_title(title, fontsize=32, pad=60)
    ax.title.set_position([.5, 1.15])  # Move title higher

row_titles = ["M=2", "M=3", "M=4"]
row_letters = ["A", "B", "C"]

# Add row titles and letters to the left of each row
for row_idx, (title, letter) in enumerate(zip(row_titles, row_letters)):
    ax = axes[row_idx, 0]
    # Add row title
    ax.annotate(
        title,
        xy=(0, 0.5),
        xycoords='axes fraction',
        fontsize=25,
        color='black',
        ha='right',
        va='center',
        rotation=0,
        annotation_clip=False,
        xytext=(-60, 0),
        textcoords='offset points'
    )
    # Add row letter at top left corner
    ax.annotate(
        letter,
        xy=(-1, 1.3),
        xycoords='axes fraction',
        fontsize=32,
        fontweight='bold',
        color='black',
        ha='left',
        va='top'
    )

plt.savefig("simulations.png", dpi=300, bbox_inches='tight')