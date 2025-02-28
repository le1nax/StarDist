from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def extract_bscan_from_tiff(tiff_path, bscan_index, output_path):
    try:
         # Open the TIFF file
        with Image.open(tiff_path) as img:
            # Check how many frames (B-scans) are in the TIFF
            num_frames = img.n_frames
            print(f"Number of B-scans in the TIFF: {num_frames}")

            # Ensure the requested index is valid
            if bscan_index < 0 or bscan_index >= num_frames:
                print("Invalid B-scan index. Please provide a valid index.")
                return

            # Seek to the specified B-scan
            img.seek(bscan_index)
            print(f"Extracting B-scan at index: {bscan_index}")

            # Convert the image to a numpy array
            img_array = np.array(img)
            print(img_array.size)

            # Check the data type and convert to uint8 or uint16
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                # Normalize the image to 0-255 for uint8 or 0-65535 for uint16
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())  # Normalize to 0-1
                img_array = (img_array * 255).astype(np.uint8)  # Scale to 0-255

            # Convert back to a Pillow image
            img_converted = Image.fromarray(img_array)
            print(img_converted.size)
            plt.imshow(img_converted)
            # Save the extracted B-scan as a PNG
            img_converted.save(output_path, format='PNG')
            print(f"B-scan saved as {output_path}")

    except Exception as e:
        print(f"Failed to extract B-scan: {e}")

# Main function
def main():
    # Define the path to your multi-page TIFF file
    current_directory = Path(__file__).resolve().parent.parent.parent
    tiff_path = current_directory / "messungen" / "32bit_float" / "brandhtcepi+10kHz+2D+FOV+1x2mm.tiff"   # Change to your TIFF file path
    bscan_index = 0  # Index of the B-scan you want to extract
    output_path = current_directory / f"bscan_{bscan_index}.png"  # Output PNG file path

    # Extract the specified B-scan
    extract_bscan_from_tiff(tiff_path, bscan_index, output_path)

if __name__ == "__main__":
    main()