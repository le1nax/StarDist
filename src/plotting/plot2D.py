import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path

# Function to plot slices with mask and clusters
def plot_slice_with_clusters(slice_img, cell_coords, labels, cell_mask):
    plt.figure(figsize=(12, 6))
    
    # Original slice
    plt.subplot(1, 3, 1)
    plt.imshow(slice_img, cmap='gray')
    plt.title("Original Slice")
    
    # Mask
    plt.subplot(1, 3, 2)
    plt.imshow(slice_img, cmap='gray')
    plt.imshow(cell_mask, cmap='binary', alpha=0.5)  # Overlay mask in red
    plt.title("Thresholded Mask")

    # DBSCAN Clusters
    plt.subplot(1, 3, 3)
    plt.imshow(slice_img, cmap='gray')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(set(labels)) - (1 if -1 in labels else 0)))
    for label, color in zip(set(labels), colors):
        if label == -1:
            continue  # Skip noise
        mask = labels == label
        plt.scatter(cell_coords[mask, 1], cell_coords[mask, 0], s=5, color=color, label=f'Cluster {label}')
    plt.title("DBSCAN Clusters")
    plt.legend()
    plt.show()