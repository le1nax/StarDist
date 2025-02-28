import numpy as np
import os
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom

def load_patches(patches_dir, base_filename="patch"):
    """
    Loads 3D patches saved as .tif files and returns them in the correct order.

    Parameters:
    - patches_dir (str): Directory where the patches are stored.
    - base_filename (str): Base name for each patch file.

    Returns:
    - patches (list of numpy arrays): List of patches in the correct order.
    """
    patches = []

    for i in range(1, 13):  # There are 12 patches
        filename = os.path.join(patches_dir, f"{base_filename}_{i:03d}.tiff")
        patch = tiff.imread(filename)
        patches.append(patch)

    return patches

def stitch_patches(patches):
    """
    Stitches 12 patches (4x3 grid) together to reconstruct the original image.

    Parameters:
    - patches (list of numpy arrays): List of 12 patches, each of shape (64, 128, 128).

    Returns:
    - original_image (numpy array): Reconstructed 3D image of shape (64, 512, 512).
    """
    if len(patches) != 12:
        raise ValueError("There must be exactly 12 patches to stitch together.")

    # Create an empty array to hold the reconstructed image
    reconstructed_image = np.zeros((64, 512, 384), dtype=patches[0].dtype)

    # Stitch the patches back together in a 4x3 grid
    patch_idx = 0
    for col in range(2, -1, -1):  # Columns 4 to 2 (reverse order)
        for row in range(4):  # All 4 rows
            row_start = row * 128
            row_end = (row + 1) * 128
            col_start = col * 128
            col_end = (col + 1) * 128

            # Place the patch in the reconstructed image
            reconstructed_image[:, row_start:row_end, col_start:col_end] = patches[patch_idx]
            patch_idx += 1

    pad_size=500
    pad_value=0
    padded_image = np.pad(
        reconstructed_image,
        pad_width=((0, 0), (pad_size, pad_size), (pad_size, pad_size)),  # No padding in the first dimension
        mode='constant',
        constant_values=pad_value
    )
    return padded_image


def overlay_3d_image_with_mask(image_path, mask_path, output_path, alpha=0.8):
    """
    Overlays a 3D zxy image stack with its same-sized mask stack using transparency.
    The output will be a grayscale uint16 image stack.

    Parameters:
    - image_path (str): Path to the input 3D image file (zxy TIFF).
    - mask_path (str): Path to the mask 3D image file (zxy TIFF).
    - output_path (str): Path to save the resulting overlayed 3D image (zxy TIFF).
    - alpha (float): Transparency level for the mask (0 = fully transparent, 1 = fully opaque).
    """
    # Load the 3D image and mask
    image = tiff.imread(image_path)  # Shape: (z, x, y) or (z, x, y, 1)
    mask = tiff.imread(mask_path)  # Shape: (z, x, y)


    # Ensure the image and mask have compatible dimensions
    if image.shape[:3] != mask.shape:
        raise ValueError("Image and mask dimensions do not match.")
    
    # If image is grayscale but has a single channel, expand it to (z, x, y, 1)
    if image.ndim == 3:  # Grayscale input (z, x, y)
        image = np.expand_dims(image, axis=-1)  # Shape: (z, x, y, 1)

    # Normalize image to [0, 1] if it's in the range [0, 65535] (16-bit)
    image = image.astype(np.float32) / 65535.0  # Normalize the image to range [0, 1]
    mask = np.where(mask > 0, 50000, 0).astype(np.float32)

    # Normalize mask to [0, 1]
    mask_normalized = mask / 65535.0 # Normalize based on the max value of the mask
    
    # Debugging: Check mask normalization
    print(f"Mask normalized min/max: {mask_normalized.min()}/{mask_normalized.max()}")

    # Expand the mask to match the image shape (z, x, y, 1)
    mask_normalized = np.expand_dims(mask_normalized, axis=-1)  # Shape: (z, x, y, 1)

    # Blend the image and the mask slice-by-slice
    overlay = (1 - alpha) * image + alpha * mask_normalized
    overlay = np.clip(overlay * 65535, 0, 65535).astype(np.uint16)  # Rescale to [0, 65535] and convert to uint16
    # Debugging: Check mask normalization
    print(f"overlay normalized min/max: {overlay.min()}/{overlay.max()}")
    print(f"shape: {overlay.shape}")

    # Save the result as a new grayscale TIFF stack (uint16)
    tiff.imwrite(output_path, overlay)
    print(f"Overlayed 3D image saved to {output_path}")

def save_reconstructed_image(image, output_path):
    """
    Saves the reconstructed image as a .tif file.

    Parameters:
    - image (numpy array): Reconstructed 3D image.
    - output_path (str): Path to save the reconstructed image.
    """
    tiff.imwrite(output_path, image.astype(np.uint16))
    print(f"Reconstructed image saved to {output_path}")

def main():
    # Define the input directory and output file path
    current_directory = Path(__file__).resolve().parent.parent.parent
    patches_dir_images = current_directory / "ML_TrainingData" / "StarDist3D" / "Training_Images"
    output_path_images = current_directory / "ML_TrainingData" / "StarDist3D" / "reconstructed_image.tiff"

    patches_dir_masks = current_directory / "ML_TrainingData" / "StarDist3D" / "Training_Masks"
    output_path_masks = current_directory / "ML_TrainingData" / "StarDist3D" / "reconstructed_masks.tiff"

    output_path_overlay = current_directory / "ML_TrainingData" / "StarDist3D" / "overlay.tiff"


    # Load the patches
    patches_images = load_patches(patches_dir_images)
    patches_masks = load_patches(patches_dir_masks)

    # Stitch the patches together
    reconstructed_image = stitch_patches(patches_images)
    reconstructed_mask = stitch_patches(patches_masks)

    # Save the reconstructed image
    save_reconstructed_image(reconstructed_image, output_path_images)
    save_reconstructed_image(reconstructed_mask, output_path_masks)
    overlay_3d_image_with_mask(output_path_images, output_path_masks, output_path_overlay)

if __name__ == "__main__":
    main()
