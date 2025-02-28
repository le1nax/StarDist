from PIL import Image
from pathlib import Path
import sys
import os


def get_project_dir():
    # Get the current directory (the directory of main_script.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Append the 'plotting' directory to sys.path
    project_dir = os.path.join(current_dir, '..')
    sys.path.append(project_dir)
    return project_dir

def createOutputPath(output_directory, filename="output.tiff"):
    # Ensure output_directory is a Path object and exists
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

     # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Create an empty (white) image
    img = Image.new("RGB", (1, 1), color="white")
    
    # Specify the full path for the output file
    output_path = os.path.join(output_directory, filename)

    # Save as TIFF file using Pillow without ImageIO
    img.save(output_path, format="TIFF")
    print(f"Empty TIFF file created at {output_path}")

    return output_path

def save_tiff_stack(tiff_stack, output_path):
    """
    Saves a 3D numpy array as a multi-page TIFF file with each slice as an 8-bit unsigned integer image.
    
    :param tiff_stack: 3D numpy array (shape: [num_slices, height, width]) to save as TIFF.
    :param output_path: File path where the TIFF stack will be saved.
    """
    # Ensure the stack is in uint8 format
    #tiff_stack = tiff_stack.astype(np.uint8)

     # Check if the output path exists; if not, create it
    if not os.path.exists(os.path.dirname(output_path)):
        createOutputPath(os.path.dirname(output_path))
    
    # Convert each 2D slice to an image
    slices = [Image.fromarray(tiff_stack[i]) for i in range(tiff_stack.shape[0])]
    print(output_path)
    
    # Save the 3D stack as a new TIFF file
    slices[0].save(output_path, save_all=True, append_images=slices[1:], compression="tiff_deflate")
    
    print(f"TIFF stack saved as: {output_path}")
