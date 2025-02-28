import os
import numpy as np
from pathlib import Path
import tifffile as tiff

import sys

# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the 'plotting' directory to sys.path
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

def whiteout_slices(image_dir, lut, save_dir):
    """
    Whites out pixel values in images starting from the given slice index.
    
    Parameters:
        image_dir (str): Path to the directory containing images.
        lut (dict): Lookup table with image index as key and slice index as value.
        save_dir (str): Path to save modified images.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for img_index, slice_index in lut.items():
        img_name = f"patch_{img_index:03d}.tiff"  # Assuming TIFF format
        img_path = os.path.join(image_dir, img_name)
        save_path = os.path.join(save_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Skipping {img_name}: File not found.")
            continue
        
        # Load image (assuming grayscale; adjust for color if needed)
        img = tiff.imread(img_path)
        
        if img is None:
            print(f"Skipping {img_name}: Unable to read image.")
            continue
        
        # Determine white pixel value based on image depth
        white_value = 255 if img.dtype == np.uint8 else 65535
        
        # Set pixels to white from the slice index onward
        img[slice_index:, :] = white_value
        
        # Save the modified image
        tiff.imwrite(save_path, img)
        print(f"Processed {img_name}, whiteout from index {slice_index}.")



def main():
    # Load the TIFF file
    current_directory = Path(__file__).resolve().parent.parent.parent
    image_dir = current_directory / "ML_TrainingData" / "StarDist3D" / "Training_Data_Reiner" / "Training_Masks"
    #input_path_tiffImage = current_directory / "output_images" / "aligned_stacks" / "Greiner_Cut_Smooth_LUT.tiff"
    save_dir = current_directory / "ML_TrainingData" / "StarDist3D" / "masks_new_whited"
    # Example usage
    lut = {1: 41, 2: 31, 3: 21, 4: 14, 5: 42, 6: 44, 7: 31, 8: 32, 10: 48, 11: 41, 12: 39}
    whiteout_slices(image_dir, lut, save_dir)

if __name__ == "__main__":
    main()