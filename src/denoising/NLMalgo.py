import cv2
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import os
from pathlib import Path


def print_timestamp(message):  # function to print timestamps
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{current_time}] {message}")


def save_image(image, path):  # check data type and convert to uint8 or uint16
    if image.dtype == np.float32 or image.dtype == np.float64:  # normalize image to 0-255 for uint8 or 0-65535 for uint16
        image = (image - image.min()) / (image.max() - image.min())  # normalize to 0-1
        image = (image * 255).astype(np.uint8)  # scale to 0-255

    img_converted = Image.fromarray(image)  # convert back to pillow image
    img_converted.save(path, format='PNG')  # save extracted B-scan as PNG
    print(f"B-scan saved as {path}")


def extract_bscan_from_tiff(tiff_path, bscan_index):
    try:
        with Image.open(tiff_path) as img:  # open tiff
            num_frames = img.n_frames  # how many frames (B-scans) are in the TIFF?
            print(f"Number of B-scans in the TIFF: {num_frames}")

            if bscan_index < 0 or bscan_index >= num_frames:  # ensure requested index is valid
                print("Invalid B-scan index. Please provide a valid index.")
                return
            img.seek(bscan_index)  # seek to specified B-scan
            print(f"Extracting B-scan at index: {bscan_index}")

            extracted_image = np.array(img)  # convert image to a numpy array
            img_array = extracted_image
            print(img_array.size)

    except Exception as e:
        print(f"Failed to extract B-scan: {e}")

    return extracted_image


def convert_image_to_int8(image):
    if image.dtype != np.uint8:
        print("Converting image to uint8...")
        image = (image - image.min()) / (image.max() - image.min())  # normalize to range [0, 1]
        image = (image * 255).astype(np.uint8)  # scale to range [0, 255] and convert to uint8
    return image


def apply_non_local_means(image, h=10, templateWindowSize=7, searchWindowSize=21):
    print_timestamp("Applying Non-Local Means denoising")
    print(image.shape)
    image = convert_image_to_int8(image)

    denoised_image = cv2.fastNlMeansDenoising(
        image,
        None,
        h,
        templateWindowSize,
        searchWindowSize
    )

    print_timestamp("Denoising completed")
    return denoised_image



def display_images(original, denoised):
    print_timestamp("Displaying original and denoised images")
    #convert the images to rgb
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_rgb)
    plt.title('Denoised Image')
    plt.axis('off')

    plt.show()
    input("Press Enter to close the images and continue...")

    print_timestamp("Images displayed successfully")


def main():
    print_timestamp("Script started")
    
    #extract B-scan from tiff and save as PNG
    current_directory = Path(__file__).resolve().parent.parent
    tiff_path = current_directory / "messungen" / "32bit_float" / "brandhtcepi+10kHz+2D+FOV+1x2mm.tiff"   # Change to your TIFF file path
    bscan_index = 0  # Index of the B-scan
    output_path_originalImage = current_directory / "output_images" / f"input_bscan_{bscan_index}.png"  # Output PNG file path
    output_path_denoisedImage = current_directory / "output_images" / f"denoised_bscan_{bscan_index}.png"

    extracted_image = extract_bscan_from_tiff(tiff_path, bscan_index)

    save_image(extracted_image, output_path_originalImage)

    #apply Non-Local Means Denoising
    denoised_image = apply_non_local_means(extracted_image)

    save_image(denoised_image, output_path_denoisedImage)

    display_images(extracted_image, denoised_image)
    print_timestamp("Script completed")


if __name__ == "__main__":
    main()



    