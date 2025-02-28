import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import scipy.ndimage as ndi
import pickle
import numpy as np
import cv2

import os
import sys


# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the 'plotting' directory to sys.path
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

from histogram_analysis.LUT import apply_lut


def moore_boundary_3d_thicc(mask, thickness=2):
    """
    Extracts the outer boundary of a 3D binary mask using Moore's boundary tracing (slice-by-slice)
    and makes the boundaries 1 pixel thicker.
    
    Parameters:
        mask (numpy.ndarray): 3D binary mask (1 for objects, 0 for background)
        thickness (int): Thickness of the boundary lines
    
    Returns:
        boundaries (numpy.ndarray): 3D mask containing only the thickened outer boundaries
    """
    boundaries = np.zeros_like(mask, dtype=np.uint8)

    # Process each 2D slice separately
    for z in range(mask.shape[0]):
        slice_mask = mask[z].astype(np.uint8)  # Get the current 2D slice
        contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw boundaries with increased thickness
        cv2.drawContours(boundaries[z], contours, -1, 1, thickness=thickness)
    
    return boundaries

def extract_outer_boundaries(mask, structure_size=1):
    """
    Extracts the outer boundaries of a 3D mask.
    
    Parameters:
        mask (numpy.ndarray): 3D binary mask (1 for objects, 0 for background)
        structure_size (int): Size of the structuring element (larger values remove thicker layers)
    
    Returns:
        boundaries (numpy.ndarray): 3D mask containing only the outer boundaries
    """
    struct_element = ndi.generate_binary_structure(3, 1)  # 3D connectivity
    eroded_mask = ndi.binary_erosion(mask, structure=struct_element, iterations=structure_size)
    boundaries = mask.astype(np.uint8) - eroded_mask.astype(np.uint8)
    return boundaries

def extract_outer_boundaries_gradient(mask):
    """
    Extracts the outer boundaries of a 3D mask using a gradient filter.
    
    Parameters:
        mask (numpy.ndarray): 3D binary mask (1 for objects, 0 for background)
    
    Returns:
        boundaries (numpy.ndarray): 3D mask containing only the outer boundaries
    """
    gradient = ndi.morphological_gradient(mask.astype(np.uint8), size=(3,3,3))  # Edge detection
    return gradient

def show_masks(test_input, labels, save_figure=False):

        print(labels.shape)

        #test_input = apply_lut(test_input)

        labels = moore_boundary_3d_thicc(labels)
        labels = np.where(labels > 0, 1, 0).astype(np.uint8)

        # Convert pixel values to 0 or 255
        test_prediction_0_to_255 = (labels > 0).astype(np.uint8) * 255

        # Get image dimensions
        Image_Z = test_input.shape[0]

        slice_idx = 36  # Start with slice ..

        # Normalize input image
        norm_input = mcolors.Normalize(vmin=np.percentile(test_input, 1), vmax=np.percentile(test_input, 99))
        norm_pred = mcolors.Normalize(vmin=np.percentile(test_prediction_0_to_255, 1), vmax=np.percentile(test_prediction_0_to_255, 99))
        norm_mask = mcolors.Normalize(vmin=np.percentile(test_prediction_0_to_255, 1), vmax=np.percentile(test_prediction_0_to_255, 99))

        # Set up figure and axes
        fig, axes = plt.subplots(1, 4, figsize=(24, 8))

        # Initialize plots (showing slice 25 initially)
        im_input = axes[0].imshow(test_input[slice_idx], norm=norm_input, cmap='magma', interpolation='nearest')
        
        # Overlay (Input Image and Prediction Mask) in the second axes
        alpha_mask = labels[slice_idx]  # This will be 0 or 1
        im_overlay_input = axes[1].imshow(test_input[slice_idx],norm=norm_input, cmap='magma', interpolation='None')  # Full opacity for input
        im_overlay_pred = axes[1].imshow(labels[slice_idx], interpolation='None',cmap='Blues',alpha=alpha_mask*0.6)  # Blue prediction mask (with transparency)

        # Display the prediction image in the third axis (full opacity)
        im_pred = axes[2].imshow(labels[slice_idx],norm=norm_input, cmap='Blues', interpolation='nearest')

        # Titles
        axes[0].set_title("Input")
        axes[1].set_title("Overlay")
        axes[2].set_title("Prediction")

        for ax in axes:
            ax.axis("off")

        # Add single slider for slice selection
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.02])
        slider = Slider(ax_slider, "Slice", 0, Image_Z - 1, valinit=slice_idx, valstep=1)


        def update(val):
            slice_idx = int(slider.val)

            # Update input image and ensure normalization is preserved
            im_input.set_data(test_input[slice_idx])
            im_input.set_norm(norm_input)  

            # Update overlay images
            im_overlay_input.set_data(test_input[slice_idx])
            im_overlay_input.set_norm(norm_input)  

            im_overlay_pred.set_data(test_prediction_0_to_255[slice_idx])
            alpha_mask = labels[slice_idx]  # This will be 0 or 1
            im_overlay_pred.set_alpha(alpha_mask*0.6)   

            # Update prediction mask
            im_pred.set_data(test_prediction_0_to_255[slice_idx])  
            im_overlay_pred.set_norm(norm_input) 

            # Redraw figure
            fig.canvas.draw_idle()

        # Attach slider to the update function
        slider.on_changed(update)

        if(save_figure):
             # Save the figure
            with open("saved_figure.pkl", "wb") as f:
                pickle.dump(fig, f)


        plt.show()