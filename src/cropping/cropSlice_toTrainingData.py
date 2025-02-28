import numpy as np
from scipy.ndimage import zoom
import os
import tifffile as tiff
from tifffile import imread
import sys
import skimage as io


# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the 'plotting' directory to sys.path
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

from fileLoadingSaving.loadSaveTiff import *
from reslicing.resliceStack import reslice_to_top_view
from reslicing.resliceStack import reslice_to_90_degrees_view
from reslicing.typeConversion import convert_to_dtype
from reslicing.resliceStack import reslice_to_90_degrees_view

def crop_to_patches(image):
    """
    Crops a 177x500x500 image into 9 patches of size 64x128x128 based on specified criteria.
    
    Parameters:
    - image (numpy array): Input 3D array of shape (84, 500, 500).
    
    Returns:
    - patches (list of numpy arrays): List of cropped patches, each of shape (64, 128, 128).
    """
    # Ensure the input image has the correct shape
    if image.shape != (84, 500, 500):
        print(image.shape)
        raise ValueError(f"Input image must have shape (84, 500, 500), but has value {image.shape}")
    
    # Crop the 1st dimension (depth)
    cropped_depth = image[1:65, 4:500, 4:500]  # Resulting shape (64, 500, 500), cut off 4 pixels that are padded on the top and left
    
    # Interpolate to resize 496x496 to 512x512
    scale_factor = 512 / 496
    upscaled_image = zoom(cropped_depth, (1, scale_factor, scale_factor), order=3)  # Shape (64, 512, 512)
    
    # Extract patches from the upscaled image
    patches = []
    for row in range(4):  # Iterate over 4 rows
        for col in range(4):  # Iterate over 4 columns
            # Get the 128x128 patch
            patch = upscaled_image[:, row * 128:(row + 1) * 128, col * 128:(col + 1) * 128]
            patches.append(patch)
    
    # Select patches from columns 4 to 2
    selected_patches = []
    for col in range(3, 0, -1):  # Columns 4 (rightmost) to 2
        for row in range(4):  # All 4 rows
            selected_patches.append(patches[row * 4 + col])
    
    return selected_patches

def save_patches_as_tif(patches, output_dir, base_filename="patch"):
    """
    Saves each 3D patch as a .tif file.
    
    Parameters:
    - patches (list of numpy arrays): List of patches, each of shape (64, 128, 128).
    - output_dir (str): Directory where the patches will be saved.
    - base_filename (str): Base name for each saved patch file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i, patch in enumerate(patches):
        # Construct the filename for each patch
        filename = os.path.join(output_dir, f"{base_filename}_{i+1:03d}.tiff")
        
        # Save the patch as a .tif file
        tiff.imwrite(filename, patch.astype(np.uint16))  # Ensure correct data type for TIFF
        print(f"Saved patch {i+1} to {filename}")


def resize_to_multiple(image, patch_size):
    """
    Resize the image using scipy.ndimage.zoom so that its dimensions are the nearest multiple of the patch size.
    """
    print(image.shape)
    d, h, w = image.shape
    pd, ph, pw = patch_size
    
    new_d = (d // pd + (d % pd > 0)) * pd
    new_h = (h // ph + (h % ph > 0)) * ph
    new_w = (w // pw + (w % pw > 0)) * pw
    
    zoom_factors = (new_d / d, new_h / h, new_w / w)
    resized_image = zoom(image, zoom_factors, order=1)  # Using bilinear interpolation
    return resized_image, (new_d, new_h, new_w), zoom_factors

def crop_image_to_patches(image, patch_size=(64, 128, 128)):
    """
    Crops a 3D image into non-overlapping patches of a given size.
    Ensures all patches are of equal dimensions.
    """
    d, h, w = image.shape
    pd, ph, pw = patch_size
    
    patches = []
    patch_coords = []
    
    for z in range(0, d, pd):
        for y in range(0, h, ph):
            for x in range(0, w, pw):
                patch = image[z:z+pd, y:y+ph, x:x+pw]
                patches.append(patch)
                patch_coords.append((z, y, x))
    
    return patches, patch_coords

def stitch_patches_back(patch_masks, patch_coords, image_shape, patch_size):
    """
    Stitches the segmented patches back into a full-size 3D image.
    """
    d, h, w = image_shape
    stitched_mask = np.zeros((d, h, w), dtype=np.uint16)
    
    pd, ph, pw = patch_size
    for (z, y, x), mask in zip(patch_coords, patch_masks):
        stitched_mask[z:z+pd, y:y+ph, x:x+pw] = mask
    
    return stitched_mask

def main():
    # Load the TIFF file
    current_directory = Path(__file__).resolve().parent.parent.parent
    input_path_tiffImage = current_directory / "output_images" / "aligned_stacks" / "aligned_reslice_LUT.tiff"
    #input_path_tiffImage = current_directory / "output_images" / "aligned_stacks" / "Greiner_Cut_Smooth_LUT.tiff"
    output_path = current_directory / "ML_TrainingData" / "StarDist3D" / "Training_Images"


    tiffImage = imread(input_path_tiffImage)

    tiffImage = convert_to_dtype(tiffImage, np.uint16)

    stacks = crop_to_patches(tiffImage)


    save_patches_as_tif(stacks, output_path)

if __name__ == "__main__":
    main()