
#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import scipy.special as scp
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap

np.set_printoptions(suppress=True)


def plot_heatmap(data, file_title=str(0), title=None,vmin=0,vmax=1,save=False,dpi=300, novel_ind=2):


    fig, axes = plt.subplots(1, len(data), figsize=(2*len(data),1.8))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6)
    for ai, ax, im in zip(np.arange(len(data)), axes, data):
        sns.heatmap(data=im, annot=True, annot_kws={"size":16}, cmap="viridis", cbar=False, fmt='.2f', ax=ax,vmin=vmin, vmax=vmax)
        ax.set_ylabel("s",fontsize=14, labelpad=5)
        ax.set_xlabel("r",fontsize=14, labelpad=5)
        ax.invert_yaxis()
        if ai < novel_ind-1:
            ax.set_title(fr"$p(r|s,\phi_{{{ai+1}}})$", fontsize=14, pad=10)
        else:
            ax.set_title(fr"$p(r|s,\phi^{{\mathrm{{novel}}}})$", fontsize=14, pad=10)
            # fr"$p(r|s,\phi_{{{col_idx+1}}}^{{\mathrm{{true}}}})$"
        ax.set_yticklabels(np.arange(1,3))
        ax.tick_params(left=False, bottom=False)
        # ax.tick_params(labelpad=10)

    return fig, axes



contingencies = [np.array([[0.16, 0.84], [0.83,0.17]]), np.ones([2,2])*0.5]
# contingencies = [np.array([[0.5,0.5], [0.66, 0.34]]), np.ones([2,2])*0.5]

titles = [r"$p(r|s,c,\phi_1)$", r"$p(r|s,c,\phi_2)$"]
fig, axes = plot_heatmap(contingencies,dpi=200, novel_ind=2)
fig.savefig("fig3_contingencies_1.svg", dpi=300, bbox_inches='tight')

# contingencies = [np.array([[0.13, 0.87], [0.87,0.13]]), np.array([[0.92, 0.08], [0.12,0.88]]), np.ones([2,2])*0.5]
contingencies = [np.array([[0.13, 0.87], [0.88,0.12]]), np.array([[0.91, 0.09], [0.15,0.85]]), np.ones([2,2])*0.5]
fig, axes = plot_heatmap(contingencies,dpi=300, novel_ind=3)
fig.savefig("fig3_contingencies_2.svg", dpi=300, bbox_inches='tight')

rew = np.array([1,0]) 
a = np.arange(2).repeat(100)
b = rew[a]

fig = plt.figure(figsize=(3.5,2),dpi=300)
plt.plot(np.arange(200),b,linewidth=4)
b = rew[np.flip(a)]
plt.plot(np.arange(200),b, linewidth=4)

plt.ylim([-0.05,1.1])
plt.ylabel("Active context", fontsize=14)
plt.yticks([0,1], fontsize=14)
plt.xlabel(r"trial $\tau$", fontsize=14)
plt.tick_params(labelsize=14)
# plt.legend()

fig.savefig("fig3_training.svg", dpi=300, bbox_inches='tight')


#%%

# titles = ["df_nb3_0.7_h70-100000.csv", "df_nb4_0.7_h70-100000.csv"]
# titles = ["df_nb4_0.7_h100000_temp.csv"]
# for title in titles:
#     df = pd.read_csv(title)
#     # df["context"] += 1
#     print(df.head())
#     cols = list(df.columns)[1:]
#     df[cols].to_csv(title, index=False)
#%% PLOT TEMPLATE BENEFIT


titles =  ["df_nb4_0.7_h70-100000.csv", "df_nb4_0.7_h100000_temp.csv"]
dfs = [pd.read_csv(title) for title in titles]
dfs[0]["template"] = 0
dfs[1]["template"] = 1


df_big = pd.concat(dfs).query("h != 70").reset_index(drop=True)

# df_big["context"] += 1
fig, ax = plt.subplots(1,2, figsize=(8,3.2))
plt.subplots_adjust(wspace=0.4)

for i in range(2):
    try:
        sns.barplot(ax=ax[i],data=df_big.query(f"context >  1 & repeated == {i}"), x="context", y="trial", hue="template", palette="gray", edgecolor="black")
    except:
        sns.barplot(ax=ax[i],data=df_big.query(f"context >  1 & repeated == {i}"), x="context", y="trial", hue="template", palette="gray", edgecolor="black")
    ax[i].grid()
    ax[i].set_ylabel(r"Trial at which $p(c_{true}) > 0.7$", fontsize=14)
    ax[i].set_ylim([0,75])
    ax[i].set_title(f"repeated: {i}")

# fig, ax = plt.subplots(1,1)
# plt.grid()
# # sns.barplot(ax=ax,data=df_big.query("context >  1"), x="context", y="trial", hue="template", palette="gray", edgecolor="black")
# sns.barplot(ax=ax,data=df_big.query("context >  1"), x="context", y="trial", hue="template", palette="gray", edgecolor="black")

# ax.set_ylabel("Trial at which p of true context > 0.7",fontsize=14)
# ax.set_ylim([0,70])


# for temp in [0,1]:
#     fig, ax = plt.subplots(1,1)
#     plt.grid()
#     sns.barplot(ax=ax,data=df_big.query(f"template=={temp}"), x="context",y="trial", hue="repeated", edgecolor="black")
#     ax.set_title(f"Used template={temp}", fontsize=14)
#     ax.set_ylabel("Trial at which p of true context > 0.7",fontsize=14)
#     ax.set_ylim([0,70])


#%% PLOT HABIT BENEFIT BAR POTS

titles = ["df_nb3_0.7_h70-100000.csv", "df_nb4_0.7_h70-100000.csv", "df_nb4_0.7_h20-100000.csv"]
df_big = pd.read_csv(titles[-1])
# df_big["context"] += 1
fig, ax = plt.subplots(1,2, figsize=(8,3.2))
plt.subplots_adjust(wspace=0.4)

for i in range(2):
    sns.barplot(ax=ax[i],data=df_big.query(f"repeated == {i} & fit >= 0.9"), x="context", y="trial", hue="h", palette="gray", edgecolor="black")
    ax[i].grid()
    ax[i].set_ylabel(r"Trial at which $p(c_{true}) > 0.7$", fontsize=14)
    ax[i].set_ylim([0,95])
    ax[i].set_title(f"repeated: {i}")

# hs = df_big["h"].unique()
# fig, ax = plt.subplots(1,1)
# sns.barplot(ax=ax, data=df_big.query("context >  1"), x="h", y="trial")

# for h in hs:
#     fig, ax = plt.subplots(1,1)
#     plt.grid()
#     sns.barplot(ax=ax,data=df_big.query(f"h=={h}"), x="context",y="trial", hue="repeated", edgecolor="black")
#     ax.set_title(f"h={h}", fontsize=14)
#     ax.set_ylabel("Trial at which p of true context > 0.7",fontsize=14)
#     ax.set_ylim([0,70])


#%% PLOT CONTINGENCIES FOR MOUSE FIGURE
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



#%%  PLOT SIMULATION ENVIRONMENT SUMMARY
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
            ax.set_xlabel(f"r \n"+fr"$p(r|s,\phi_{{{col_idx+1}}}^{{\mathrm{{true}}}})$", fontsize=22)
            
            ax.set_ylabel("s", fontsize=22, labelpad=10)
            # ax.set_xlabel("r", fontsize=22, labelpad=10)

            ax.tick_params(axis='both', left=False, bottom=False, labelsize=16)
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
        ax.plot(x , y, linewidth=4)  # Placeholder line
    ax.set_xlabel(r"trial $\tau$", fontsize=20)
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_label_position('left')
    ax.spines[['right','top']].set_visible(False)
    ax.set_ylabel("Active context", fontsize=22)
    ax.set_xticks(np.arange(0,nb*100+1,100))
    ax.tick_params(axis='x', labelsize=16,rotation=45)
    ax.tick_params(axis='y', labelsize=16)

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

fig.savefig("fig1.svg", dpi=300, bbox_inches='tight')