import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, LinearSegmentedColormap

def visualize(tensor_data, figsize=(4, 4), x_labels=None, y_labels=None, title='', color='red'):
    x_edges = np.arange(tensor_data.shape[1] + 1) - 0.5
    y_edges = np.arange(tensor_data.shape[0] + 1) - 0.5

    log_norm = LogNorm(vmin=tensor_data.min(), vmax=tensor_data.max())
    if color == 'red':
        light_color = LinearSegmentedColormap.from_list('CustomColors', ['#ffffff', '#F7D0D0', '#8B0000'], N=256)
    else:
        light_color = LinearSegmentedColormap.from_list('CustomColors', ['#ffffff', '#D0E4F7', '#08306b'], N=256)

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(x_edges, y_edges, tensor_data, cmap=light_color, norm=log_norm, edgecolors='white', linewidth=2)

    ax.set_aspect('equal')
    ax.set_xticks(np.arange(tensor_data.shape[1]))
    ax.set_yticks(np.arange(tensor_data.shape[0]))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    fig.colorbar(cax, ax=ax, format='%.0e')
    fig.suptitle(title, y=0.02, fontsize=12, ha='center')

    plt.tight_layout()
    plt.show()
