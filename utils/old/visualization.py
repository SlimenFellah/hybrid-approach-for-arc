import matplotlib.pyplot as plt

def plot_grid(grid, title='Grid'):
    """Displays a 2D grid with color mapping."""
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap='tab20', vmin=0, vmax=19)  # Customize the colormap as needed
    plt.title(title)
    plt.colorbar()
    plt.show()

def compare_grids(true_output, predicted_output, task_id='Task'):
    """Compares the true and predicted outputs side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(true_output, cmap='tab20', vmin=0, vmax=19)
    axes[0].set_title(f'{task_id} - True Output')
    
    axes[1].imshow(predicted_output, cmap='tab20', vmin=0, vmax=19)
    axes[1].set_title(f'{task_id} - Predicted Output')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
