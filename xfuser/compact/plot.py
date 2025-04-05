import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

PLOT_DIR = "plots"

def plot_3d(tensor, title, filename=None):
    # Plot
    tensor = tensor.cpu()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(tensor.shape[1]), np.arange(tensor.shape[0]))
    z = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
    ax.plot_surface(x, y, z, cmap='coolwarm', linewidth=0, antialiased=False)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Token')
    ax.set_zlabel('Tenor')
    plt.title(title)
    
    # Save to file
    if filename is None:
        filename = f"{PLOT_DIR}/3d_{title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
        
    return fig, ax
