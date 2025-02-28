from skimage import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
import sys
import os

# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the 'plotting' directory to sys.path
plotting_dir = os.path.join(current_dir, '..')
sys.path.append(plotting_dir)

from plotting.plot3D import plotTiff3D
from fileLoadingSaving.loadSaveTiff import *
#from plotting.plot2D import plotTiff2D

def crop_3d_image(input_image, x_range, y_range, z_range, output_path):
    """
    Crops a 3D image (e.g., OCT data) based on specified x, y, z ranges and saves the result.

    Parameters:
    - input_image: The 3D image to be cropped (from skimage.io.imread()).
    - x_range: Tuple (start_x, end_x) for x-axis crop.
    - y_range: Tuple (start_y, end_y) for y-axis crop.
    - z_range: Tuple (start_z, end_z) for z-axis crop.
    - output_path: Path where the cropped image will be saved.
    """

    # Ensure the ranges are within the bounds of the image dimensions
    x_start, x_end = x_range
    y_start, y_end = y_range
    z_start, z_end = z_range

    # Cropping the image to the specified ranges
    cropped_image = input_image[z_start:z_end, y_start:y_end, x_start:x_end]

    # Saving the cropped image to the output path
    save_tiff_stack(cropped_image, output_path)

    print(f"Cropped image saved at {output_path}")

    return cropped_image

def validateCroppedShape(output_path_croppedImage):

    oct_data = io.imread(output_path_croppedImage)

    # Check the data type and convert if necessary
    if oct_data.dtype != np.int8:
        # Normalize the float data to fit within int8 range
        oct_data = ((oct_data - oct_data.min()) / (oct_data.max() - oct_data.min()) * 255).astype(np.int8)

    print("The resulting shape of the cropped Image is: ")
    print(oct_data.shape)
    print("It has been saved at: ", output_path_croppedImage)


def main():
    # Load the TIFF file
    current_directory = Path(__file__).resolve().parent.parent.parent
    input_path_tiffImage = current_directory / "messungen" / "3dim" / "greinermodell_lsm03.tiff"
    output_directory = current_directory / "cropped_images"

    output_path = createOutputPath(output_directory, "croppedTiffpy.tiff")

    tiffImage = io.imread(input_path_tiffImage)

    cropped_image = crop_3d_image(tiffImage, (1,100), (100,250), (1,100), output_path)

    validateCroppedShape(output_path)

    display = 1
    if display == 1 :
        # Display the cropped 3D stack slices
        #plotTiff(tiffImage)
        plotTiff3D(cropped_image)

if __name__ == "__main__":
    main()