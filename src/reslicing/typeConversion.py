import numpy as np
from pathlib import Path
import os, sys
from skimage import io

# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the 'plotting' directory to sys.path
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

from fileLoadingSaving.loadSaveTiff import *

def convert_to_dtype(image, dtype=np.uint16):
    """
    Converts an input image to the specified data type.
    
    Parameters:
    - image (numpy array): Input image to be converted.
    - dtype (numpy data type, optional): Desired data type for the output. Default is np.uint16.
    
    Returns:
    - converted_image (numpy array): Image converted to the specified data type.
    """
    # Check if the input is a numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array.")
    
    # Perform the conversion
    try:
        converted_image = image.astype(dtype)
    except TypeError:
        raise ValueError(f"Unsupported data type: {dtype}.")
    
    return converted_image

def main():
    # Load the TIFF file
    current_directory = Path(__file__).resolve().parent.parent.parent
    input_path_tiffImage = current_directory / "output_images" / "segmented_images" / "images" / "patch_002.tiff"
    #input_path_tiffImage = current_directory / "output_images" / "aligned_stacks" / "Greiner_Cut_Smooth_LUT.tiff"
    output_path = current_directory / "output_images" / "converted_images" / "rawimage_converted.tiff"

    out_image = convert_to_dtype(io.imread(input_path_tiffImage))
    print(out_image.shape)
    #save_tiff_stack(out_image, output_path)

if __name__ == "__main__":
    main()