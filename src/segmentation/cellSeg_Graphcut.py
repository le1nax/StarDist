import numpy as np
import cv2
from skimage import io, segmentation, color, graph, img_as_ubyte

import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path

from skimage.color import rgb2gray

import sys
import os

# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append subdir' directory to sys.path
sub_dir = os.path.join(current_dir, '..')
sys.path.append(sub_dir)

# Imports
try:
    from cropping.crop3DTiff_skimage import crop_3d_image
    #from plotting.plot2D import plotTiff2D
    from plotting.plot3D import plotTiff3D
    from denoising.BM3DAlgo import applyBM3D
except ImportError as e:
    print(f"Import error: {e}. Check if the directories and files are correctly structured.")


# Load the 3D OCT image (assuming a TIFF stack)
def load_3d_image(file_path):
    img_stack = io.imread(file_path)  # Load the 3D TIFF image
    return img_stack

# Load the 2D OCT image (assuming a TIFF stack)
def load_2d_image(file_path, slice=50, firstDimAxial=False):
    img_stack = io.imread(file_path)  # Load the 3D TIFF image
    if img_stack.ndim == 3:
        if(firstDimAxial):
            image = img_stack[slice, :, :]
        else: #assume 3rd dim to be depth
            image = img_stack[:, :, slice]
    elif img_stack.ndim == 2:
        image = img_stack
        print("The image is 2-dimensional.")
    else:
        print("The image has an unexpected number of dimensions: ")
        print(img_stack.ndim)
        sys.exit("Terminating script due to unexpected image dimensions.")
    
    return image


# Apply a single threshold to the image to enhance contrast
def apply_single_threshold(img, threshold):
    # Create an output image with zeros
    contrasted_img = np.zeros_like(img)
    
    # Set pixel values based on the threshold
    contrasted_img[img >= threshold] = 255  # Set to white for edges
    contrasted_img[img < threshold] = 0     # Set to black for non-edges
    
    return contrasted_img


# Process each slice in the 3D OCT image
def segment_image(img_stack):
    # Perform graph-cut segmentation
    print(img_stack.shape)
    labels = segmentation.slic(img_stack, compactness=10, n_segments=200, start_label=1, channel_axis =None)
    print(labels.shape)
    g = graph.rag_mean_color(img_stack, labels, mode='similarity')
    labels2 = graph.cut_normalized(labels, g)

    # Map the labels to colors and display the segmented image
    segmented_image = color.label2rgb(labels2, img_stack, kind='avg')
    
    return np.array(segmented_image)

def convert_image_to_uint8(image):
    
    # Ensure the image is loaded correctly
    if image is None:
        raise ValueError("Image could not be loaded. Please check the path.")
    
    # Normalize and convert to uint8, if needed
    if image.dtype == np.uint8:
        # Image is already in uint8 format
        return image
    else:
        # Scale the image to [0, 255] and convert to uint8
        image_min = image.min()
        image_max = image.max()

        # Handle the case where the image has no range (e.g., a constant image)
        if image_max == image_min:
            return np.zeros_like(image, dtype=np.uint8)

        # Normalize and scale to uint8
        image_uint8 = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
    
        return image_uint8

# Display a few slices from the original and processed stacks
def display_results(original_stack, processed_stack):
    original_stack_int8 = convert_image_to_uint8(original_stack)
    processed_stack_int8 = convert_image_to_uint8(processed_stack)
    plotTiff3D([original_stack_int8, processed_stack_int8])

# Main function to load, process, and gather images for final display
def main(input_path, output_path):
    # List to store processed images
    image_list = []

    # Load the 3D image
    oct_stack = load_2d_image(input_path, 64, True)
    print(oct_stack.dtype)
    input_image = oct_stack.astype(np.float32) / 65535.0 if oct_stack.max() > 1 else oct_stack #bm3d assumes [0,1]
    print(input_image.max())
    image_list.append(input_image)

    # Denoise
    blurred_image = cv2.medianBlur(input_image, 5)
    image_list.append(blurred_image)

    denoised_image = applyBM3D(blurred_image, 0.01)
    image_list.append(denoised_image)

    print(denoised_image.dtype)
    print(denoised_image.shape)

    # Define the threshold
    threshold = 160  # Set your desired intensity threshold here

    tresh_img = np.zeros_like(denoised_image)

    # Set values above the threshold to white (1.0)
    tresh_img[denoised_image > threshold] = 1

    print(tresh_img)

    image_list.append(tresh_img)



    # Process the 3D image
    #segmented_stack = segment_image(denoised_image)
    #image_list.append(segmented_stack)

    # Display the collected images
    plotTiff3D(image_list, is_list=True)

# Example usage
current_directory = Path(__file__).resolve().parent.parent.parent
input_path_tiffImage = current_directory / "messungen" / "3dim" / "Reslice_of_Reiner_ROI_16bit.tiff"
output_path_labeledImage = current_directory / "output_images" / "segmented_images" / "segmented_image.tiff" 

main(input_path_tiffImage, output_path_labeledImage)