import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def plot_heatmap(data, ind=str(0), title=None,vmin=0,vmax=1):
    
    if not type(data) is list:
        data = [data]
        title = [title]
        
    fig, axes = plt.subplots(1,len(data), figsize=(6*len(data), 6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    

    for ai, ax, im in zip(np.arange(len(data)), axes, data):
        g = sns.heatmap(data=im, annot=True, annot_kws={"size":16}, cmap="viridis", cbar=False, fmt='.2f', ax=ax,vmin=vmin, vmax=vmax)
        
        if title is not None:
            ax.set_title(title[ai])
        
        # ax.set_xticks(fontsize=12)
        # g.set_x_

    # plt.show()
    plt.savefig(ind + ".png")
    
    plt.close()
    # return fig, axes