import numpy as np
import cv2
from skimage import io, morphology
from sklearn.cluster import DBSCAN
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_dir, '..')
sys.path.append(project_dir)

from plotting.plot3D import plotTiff3D
from plotting.plot2D import plot_slice_with_clusters
from fileLoadingSaving.loadSaveTiff import *
from reslicing.resliceStack import reslice_to_top_view

def intensity_tresh_mask(slice_img, lower_bound, upper_bound):
    cell_mask = (slice_img >= lower_bound) & (slice_img <= upper_bound)
    return cell_mask

def closing_operation(mask, discSize=3):
    closedCellMask = morphology.binary_closing(mask, morphology.disk(discSize))
    return closedCellMask

def dbscanMask(cell_coords, dbscan_eps, min_samples):
    n_clusters_current_iteration = 0
    db = 0
    if len(cell_coords) > 0:
        db = DBSCAN(eps=dbscan_eps, min_samples=min_samples).fit(cell_coords)
        n_clusters_current_iteration = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        
    return db, n_clusters_current_iteration

def countCellsDBScan(stack, dbscanEps, min_samples, lower_bound, upper_bound, visualizeMasks=True):
    totalCellCount = 0
    # Iterate over each slice in the stack
    for slice_idx in [60, 150]: #range(stack.shape[0]):
        slice_img = stack[slice_idx]
        # Step 1: Thresholding to segment cells by intensity range
        cell_mask = intensity_tresh_mask(slice_img, lower_bound, upper_bound)
    
        # Step 2: Optional - Morphological closing to improve segmentation
        #closedCellMask = closing_operation(cell_mask)
    
        # Step 3: Get coordinates of pixels within the intensity range
        cell_coords = np.column_stack(np.where(cell_mask))
    
        # Step 4: Apply DBSCAN clustering
        db, n_clusters = dbscanMask(cell_coords, dbscanEps, min_samples)

        #vis certain slices
        if(visualizeMasks==True):
            if(slice_idx == 150):
                plot_slice_with_clusters(slice_img, cell_coords, db.labels_, cell_mask)

        totalCellCount += n_clusters

    return totalCellCount

def main():
    current_directory = Path(__file__).resolve().parent.parent.parent
    #input_path_tiffImage = current_directory / "output_images" / "aligned_stacks" / "aligned_reiner.tiff"
    input_path_tiffImage = current_directory / "messungen" / "3dim" / "Reslice_of_Reiner_ROI_16bit_postprocessed.tiff"
    stack = io.imread(input_path_tiffImage)  # shape (num_slices, height, width)
    #topview_stack = reslice_to_top_view(stack)
    #topview_stack = cv2.medianBlur(topview_stack, 5)
    # # Define intensity range for skin cells
    lower_bound = round(0)
    upper_bound = round(61943)
    print(stack.shape)

    # print(lower_bound)
    # print(upper_bound)

    # # Parameters for DBSCAN clustering
    eps = 2  # Distance threshold; adjust based on cell spacing in pixels
    min_samples = 10  # Minimum points in a cluster; adjust for cell density
    totalCellCount = countCellsDBScan(stack, eps, min_samples, lower_bound, upper_bound)
    print("Total number of skin cells:", totalCellCount)



if __name__ == "__main__":
    main()

