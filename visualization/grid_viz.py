import matplotlib.pyplot as plt
import numpy as np
from src.config import Config

class GridVisualizer:
    @staticmethod
    def visualize_grid(grid, title=None):
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='tab10', vmin=0, vmax=9)
        plt.grid(True)
        if title:
            plt.title(title)
        plt.colorbar()
        plt.show()
    
    @staticmethod
    def visualize_sample(input_grid, output_grid, prediction=None):
        fig, axes = plt.subplots(1, 3 if prediction is not None else 2, figsize=(15, 5))
        
        axes[0].imshow(input_grid, cmap='tab10', vmin=0, vmax=9)
        axes[0].set_title('Input')
        axes[0].grid(True)
        
        axes[1].imshow(output_grid, cmap='tab10', vmin=0, vmax=9)
        axes[1].set_title('Expected Output')
        axes[1].grid(True)
        
        if prediction is not None:
            axes[2].imshow(prediction, cmap='tab10', vmin=0, vmax=9)
            axes[2].set_title('Prediction')
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_attention(grid, attention_weights, title=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Original grid
        im1 = ax1.imshow(grid, cmap='tab10', vmin=0, vmax=9)
        ax1.set_title('Original Grid')
        ax1.grid(True)
        plt.colorbar(im1, ax=ax1)
        
        # Attention weights
        im2 = ax2.imshow(attention_weights, cmap='hot')
        ax2.set_title('Attention Weights')
        ax2.grid(True)
        plt.colorbar(im2, ax=ax2)
        
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        plt.show()
