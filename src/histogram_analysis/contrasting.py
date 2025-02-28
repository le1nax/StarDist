import numpy as np
from skimage import io, morphology
from sklearn.cluster import DBSCAN
from PIL import Image
from pathlib import Path
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_dir, '..')
sys.path.append(project_dir)

from plotting.plot3D import plotTiff3D
from fileLoadingSaving.loadSaveTiff import *

current_directory = Path(__file__).resolve().parent.parent.parent
input_path_tiffImage = current_directory / "output_images" / "aligned_stacks" / "aligned_reiner.tiff"
# Load your OCT stack (assuming a 3D numpy array)
stack = io.imread(input_path_tiffImage)  # shape (num_slices, height, width)

# Define intensity range for skin cells
lower_bound = round(61243 / 255)
upper_bound = round(61616 / 255)

# Initialize cell count
total_cell_count = 0

# Parameters for DBSCAN clustering
eps = 30  # Distance threshold; adjust based on cell spacing in pixels
min_samples = 10  # Minimum points in a cluster; adjust for cell density

# Iterate over each slice in the stack
for slice_idx in range(stack.shape[0]):
    slice_img = stack[slice_idx]
    
    # Step 1: Thresholding to segment cells by intensity range
    cell_mask = (slice_img >= lower_bound) & (slice_img <= upper_bound)
    
    # Step 2: Optional - Morphological closing to improve segmentation
    cell_mask = morphology.binary_closing(cell_mask, morphology.disk(3))
    
    # Step 3: Get coordinates of pixels within the intensity range
    cell_coords = np.column_stack(np.where(cell_mask))
    
    # Step 4: Apply DBSCAN clustering
    if len(cell_coords) > 0:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(cell_coords)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        total_cell_count += n_clusters

print("Total number of skin cells:", total_cell_count)