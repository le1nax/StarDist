from pystackreg import StackReg
from skimage import io, registration
from skimage.transform import warp
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def register_stack(input_stack, method='mean'):
    sr = StackReg(StackReg.RIGID_BODY)
    registered_stack = sr.register_transform_stack(input_stack, reference=method, axis=0)
    
    return registered_stack

def debug_dense_register_stack(image_stack):
    slice_0 = image_stack[4]
    slice_1 = image_stack[5]

    # Calculate optical flow for a single slice pair
    flow_y, flow_x = registration.optical_flow_tvl1(slice_0, slice_1)

    # Check flow vector range
    print("Flow Y min, max:", np.min(flow_y), np.max(flow_y))
    print("Flow X min, max:", np.min(flow_x), np.max(flow_x))

    # Try warping the second slice with the flow
    coords = np.meshgrid(np.arange(slice_0.shape[0]), np.arange(slice_0.shape[1]), indexing='ij')
    
    displaced_coords = np.array([coords[0] + flow_y, coords[1] + flow_x])
    warped_slice = warp(slice_1, displaced_coords, mode='edge')

    # Visualize the result
    plt.imshow(warped_slice, cmap='gray')
    plt.show()

def dense_register_stack_bilateral(image_stack, axis=0) :
    """
    Apply dense registration to align an image stack.

    Parameters:
    - image_stack (numpy.ndarray): 3D numpy array of shape (num_slices, height, width).
    - axis (numerical value): axis index that defines the direction of the aligned slices
    Returns:
    - registered_stack (numpy.ndarray): Registered 3D numpy array of the same shape as input.
    """
    image_stack = image_stack.astype(np.float32) / 65535.0

    registered_stack = np.copy(image_stack)
    num_slices = image_stack.shape[axis]

    if(axis==0):
        for i in range(1, num_slices):
            # Calculate the optical flow from slice i-1 to slice i
            flow_y, flow_x = registration.optical_flow_tvl1(registered_stack[i - 1], registered_stack[i])
            # Create coordinate grid for warping
            coords = np.meshgrid(np.arange(registered_stack.shape[1]), np.arange(registered_stack.shape[2]), indexing='ij')
            displaced_coords = np.array([coords[0] + flow_y, coords[1] + flow_x])
            # Ensure that displaced coordinates are within bounds
            displaced_coords = np.clip(displaced_coords, 0, np.array([registered_stack.shape[1] - 1, registered_stack.shape[2] - 1])[:, None, None])
            # Warp the current slice to align it with the previous slice
            warped_slice = warp(registered_stack[i], displaced_coords, mode='edge')
            # Update only after warping succeeds
            registered_stack[i] = warped_slice

    elif(axis==1):
        for i in range(1, num_slices):
            # Calculate the optical flow from slice i-1 to slice i
            flow_y, flow_x = registration.optical_flow_tvl1(registered_stack[:,i - 1,:], registered_stack[:,i,:])
            # Create coordinate grid for warping
            coords = np.meshgrid(np.arange(registered_stack.shape[0]), np.arange(registered_stack.shape[2]), indexing='ij')
            displaced_coords = np.array([coords[0] + flow_y, coords[1] + flow_x])
            # Ensure that displaced coordinates are within bounds
            displaced_coords = np.clip(displaced_coords, 0, np.array([registered_stack.shape[0] - 1, registered_stack.shape[2] - 1])[:, None, None])
            # Warp the current slice to align it with the previous slice
            warped_slice = warp(registered_stack[:,i,:], displaced_coords, mode='edge')
            # Update only after warping succeeds
            registered_stack[:,i,:] = warped_slice
    elif(axis==2):
        for i in range(1, num_slices):
            # Calculate the optical flow from slice i-1 to slice i
            flow_y, flow_x = registration.optical_flow_tvl1(registered_stack[:,:,i - 1], registered_stack[:,:,i])
            # Create coordinate grid for warping
            coords = np.meshgrid(np.arange(registered_stack.shape[0]), np.arange(registered_stack.shape[1]), indexing='ij')
            displaced_coords = np.array([coords[0] + flow_y, coords[1] + flow_x])
            # Ensure that displaced coordinates are within bounds
            displaced_coords = np.clip(displaced_coords, 0, np.array([registered_stack.shape[0] - 1, registered_stack.shape[1] - 1])[:, None, None])
            # Warp the current slice to align it with the previous slice
            warped_slice = warp(registered_stack[:,:,i], displaced_coords, mode='edge')
            # Update only after warping succeeds
            registered_stack[:,:,i]= warped_slice
    else:
            print("Axis out of bounce. Returning input image")
            return image_stack


    return registered_stack


def dense_register_stack(image_stack) :
    """
    Apply dense registration to align an image stack.

    Parameters:
    - image_stack (numpy.ndarray): 3D numpy array of shape (num_slices, height, width).

    Returns:
    - registered_stack (numpy.ndarray): Registered 3D numpy array of the same shape as input.
    """
    image_stack = image_stack.astype(np.float32) / 65535.0

    registered_stack = np.copy(image_stack)
    num_slices = image_stack.shape[0]

    for i in range(1, num_slices):

        # Calculate the optical flow from slice i-1 to slice i
        flow_y, flow_x = registration.optical_flow_tvl1(registered_stack[i - 1], registered_stack[i])

        # Create coordinate grid for warping
        coords = np.meshgrid(np.arange(registered_stack.shape[1]), np.arange(registered_stack.shape[2]), indexing='ij')
        
        displaced_coords = np.array([coords[0] + flow_y, coords[1] + flow_x])

        # Ensure that displaced coordinates are within bounds
        displaced_coords = np.clip(displaced_coords, 0, np.array([registered_stack.shape[1] - 1, registered_stack.shape[2] - 1])[:, None, None])

        # Warp the current slice to align it with the previous slice
        warped_slice = warp(registered_stack[i], displaced_coords, mode='edge')

        # Update only after warping succeeds
        registered_stack[i] = warped_slice
        #print(warped_slice)
        #print(registered_stack[i])

    return registered_stack
# # register each frame to the previous (already registered) one
# # this is what the original StackReg ImageJ plugin uses
# out_previous = sr.register_transform_stack(img0, reference='previous')

# # register to first image
# out_first = sr.register_transform_stack(img0, reference='first')

# # register to mean image
# out_mean = sr.register_transform_stack(img0, reference='mean')MN

# # register to mean of first 10 images
# out_first10 = sr.register_transform_stack(img0, reference='first', n_frames=10)

# # calculate a moving average of 10 images, then register the moving average to the mean of
# # the first 10 images and transform the original image (not the moving average)
# out_moving10 = sr.register_transform_stack(img0, reference='first', n_frames=10, moving_average = 10)

def main():
    # Load the TIFF file
    current_directory = Path(__file__).resolve().parent.parent.parent
    input_path_tiffImage = current_directory / "messungen" / "3dim" / "Greiner_Cut_16bit.tif"
    #input_path_tiffImage = current_directory / "output_images" / "aligned_stacks" / "Greiner_Cut_Smooth_LUT.tiff"
    output_path = current_directory / "output_images" / "aligned_stacks" / "aligned_reslice_raw.tiff"

    tiffImage = io.imread(input_path_tiffImage)
   # tiffImage = cv2.medianBlur(tiffImage, ksize=3)
    print(tiffImage.shape)
    #tiffImage = convert_to_dtype(tiffImage)

    
    #tiffImage = reslice_to_90_degrees_view(tiffImage, 'y')
    #registered_stack = tiffImage
    registered_stack = dense_register_stack_bilateral(tiffImage, 0)
    reslice = reslice_to_90_degrees_view(registered_stack)

    reslice = reslice * 65536 #convert to float


    registered_stack = dense_register_stack_bilateral(reslice, 0)
    registered_stack = registered_stack * 65536
    #registered_stack = reslice_to_90_degrees_view(registered_stack, 'y')

    save_tiff_stack(registered_stack, output_path)

    tiffImage = reslice_to_90_degrees_view(tiffImage, 'y')
    registered_stack = reslice_to_90_degrees_view(registered_stack, 'y')

    image_list = []
    image_list.append(tiffImage[:, :, 1])
    image_list.append(tiffImage[:, :, 100])
    image_list.append(tiffImage[:, :, 200])
    image_list.append(tiffImage[:, :, 300])
    image_list.append(tiffImage[:, :, 400])
    image_list.append(tiffImage[:, :, 499])
    image_list.append(registered_stack[:, :, 1])
    image_list.append(registered_stack[:, :, 100])
    image_list.append(registered_stack[:, :, 200])
    image_list.append(registered_stack[:, :, 300])
    image_list.append(registered_stack[:, :, 400])
    image_list.append(registered_stack[:, :, 499])

    plotTiff3D(image_list, True)


if __name__ == "__main__":
    main()