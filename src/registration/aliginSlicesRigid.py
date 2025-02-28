import numpy as np
from scipy.ndimage import sobel, shift, median_filter, gaussian_filter
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage import zoom

from skimage import io

import sys
import os

# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the 'plotting' directory to sys.path
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

from fileLoadingSaving.loadSaveTiff import *
from plotting.plot3D import plotTiff3D
from reslicing.resliceStack import reslice_to_90_degrees_view
from reslicing.typeConversion import convert_to_dtype
 
def median_filter(arr):
    """
    Applies a median filter to a 1D array, replacing each element with the median of itself,
    the two preceding elements, and the two following elements. Handles edge cases by
    considering only available neighbors with zero padding.
    
    Parameters:
    arr (list or np.ndarray): Input array.
    
    Returns:
    np.ndarray: Filtered array.
    """
    arr = np.asarray(arr)
    n = len(arr)
    padded_arr = np.pad(arr, (2, 2), mode='constant', constant_values=0)
    filtered = np.zeros_like(arr)
    
    for i in range(n):
        window = padded_arr[i : i + 10]  # Extract relevant window
        filtered[i] = np.median(window)
    
    return filtered



# Function to find the index of the upper skin layer using Gaussian smoothing and Sobel edge detection
def find_upper_layer_index_with_sobel(column_data):
   
    #column_data = median_filter(column_data)
    non_black_indices = np.nonzero(column_data)[0]
    if non_black_indices.size > 0:
        return non_black_indices[0]
    else:
        return -1  # Return -1 if all pixels are black

# Function to find the index of the upper skin layer using Gaussian smoothing and Sobel edge detection
def find_lower_layer_index_with_sobel(column_data):
   
    #column_data = median_filter(column_data)
    non_black_indices = np.nonzero(column_data)[0]
    if non_black_indices.size > 0:
        return non_black_indices[-1]
    else:
        return -1  # Return -1 if all pixels are black
    

def scale_column_to_reference_length(column_data, ref_length):
    """
    Scale a column to match the reference length using zoom to preserve intensity.
    
    Parameters:
    column_data (np.ndarray): The input column data.
    ref_length (int): The target reference length.
    
    Returns:
    np.ndarray: Scaled column data.
    """
    non_black_indices = np.nonzero(column_data)[0]
    if len(non_black_indices) < 2:
        return column_data  # Skip if the column is empty or too short
    
    upper_idx = non_black_indices[0]
    lower_idx = non_black_indices[-1]
    current_length = lower_idx - upper_idx

    if current_length <= 0:
        return column_data  # Avoid division by zero if invalid
    
    # Extract the nonzero part of the column
    nonzero_part = column_data[upper_idx:lower_idx+1]

    # Compute zoom factor
    zoom_factor = ref_length / len(nonzero_part)
    
    # Apply zoom
    scaled_nonzero_part = zoom(nonzero_part, zoom_factor, order=1)  # Linear interpolation

    # Reconstruct the full column with adjusted size
    new_column = np.zeros_like(column_data)
    new_lower_idx = upper_idx + len(scaled_nonzero_part) - 1
    new_column[upper_idx:new_lower_idx + 1] = scaled_nonzero_part  # Fill scaled part

    return new_column

# Function to move the column to the reference index
def move_column_to_index(column_data, reference_index, target_index):
    """
    Shift the column to align with the reference column based on the detected upper layer.
    :param column_data: 1D numpy array representing a single column (along z).
    :param reference_index: The index of the reference column's upper skin layer.
    :param target_index: The index of the target column's upper skin layer.
    :return: The shifted column data.
    """
    shift_value = reference_index - target_index  # Calculate how much to shift
    shifted_column = shift(column_data, shift_value)  # Shift the column accordingly
    newidx = find_upper_layer_index_with_sobel(shifted_column)
    #print(f"new column hight is {newidx}")
    #print(f"image has been shifted by {shift_value}")
    
    return shifted_column

def align_tz_planes(image):
    """
    Align the t, z planes of the image along the x-axis by processing each column.
    Also scales columns to ensure uniform length.
    
    :param image: 3D numpy array (t, z, x).
    :return: Aligned and scaled 3D image.
    """
    aligned_image = np.copy(image)
    column_reference = image[0, :, 0]  # First column as reference
    
    upper_index_reference_column = find_upper_layer_index_with_sobel(column_reference)
    lower_index_reference_column = find_lower_layer_index_with_sobel(column_reference)
    ref_length = lower_index_reference_column - upper_index_reference_column  # Reference length
    
    # Iterate over x-axis
    for x in range(image.shape[2]):
        for t in range(image.shape[0]):
            column_to_be_aligned = image[t, :, x]

            # Align to reference index
            upper_index_column_to_be_aligned = find_upper_layer_index_with_sobel(column_to_be_aligned)
            aligned_column = move_column_to_index(column_to_be_aligned, upper_index_reference_column, upper_index_column_to_be_aligned)

            # Scale column to match reference length
            scaled_column = scale_column_to_reference_length(aligned_column, ref_length)
            
            aligned_image[t, :, x] = scaled_column

    return aligned_image

def main():
    # Load the TIFF file
    current_directory = Path(__file__).resolve().parent.parent.parent
    input_path_tiffImage = current_directory / "messungen" / "32bit_float" / "masked_16bit.tif"
    output_path = current_directory / "output_images" / "aligned_stacks" / "addi_align.tiff"

    tiffImage = io.imread(input_path_tiffImage)
    #tiffImage = tiffImage[0:300, 0:150, 0:800]
    # tiffImage = cv2.medianBlur(tiffImage, ksize=3)
    aligned_xy = align_tz_planes(tiffImage)

    print(tiffImage.shape)
    # registered_stack = dense_register_stack_bilateral(reslice, 0)
    # registered_stack = registered_stack * 65536
    #registered_stack = reslice_to_90_degrees_view(registered_stack, 'y')

    save_tiff_stack(aligned_xy, output_path)

    tiffImage = reslice_to_90_degrees_view(tiffImage, 'y')
    aligned_xy = reslice_to_90_degrees_view(aligned_xy, 'y')

    image_list = []
    image_list.append(tiffImage[:, :, 1])
    # image_list.append(tiffImage[:, :, 100])
    # image_list.append(tiffImage[:, :, 200])
    # # image_list.append(tiffImage[:, :, 300])
    # # image_list.append(tiffImage[:, :, 400])
    # # image_list.append(tiffImage[:, :, 499])
    image_list.append(aligned_xy[:, :, 1])
    # image_list.append(aligned_xy[:, :, 100])
    # image_list.append(aligned_xy[:, :, 200])
    # # image_list.append(aligned_xy[:, :, 300])
    # # image_list.append(aligned_xy[:, :, 400])
    # # image_list.append(aligned_xy[:, :, 499])

    plotTiff3D(image_list, True)


if __name__ == "__main__":
    main()