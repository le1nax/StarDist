import torchio as tio
import numpy as np
import torch
from pathlib import Path
from skimage import io
import matplotlib.pyplot as plt

import sys
import os

# Get the current directory (the directory of main_script.py)
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# Append the 'plotting' directory to sys.path
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

from fileLoadingSaving.loadSaveTiff import *
from reslicing.resliceStack import reslice_to_90_degrees_view


def main():
    # Define the input directory with image and label pairs
    img_input_dir = current_dir / "Training_Images"
    mask_input_dir = current_dir / "Training_Masks"
    image_files = sorted(img_input_dir.glob("*.tiff"))
    label_files = sorted(mask_input_dir.glob("*.tiff"))

    # Define output directories
    output_dir_img = current_dir / "augmented_images"
    output_dir_mask = current_dir / "augmented_masks"
    output_dir_img.mkdir(parents=True, exist_ok=True)
    output_dir_mask.mkdir(parents=True, exist_ok=True)

    # Iterate through image-label pairs
    for image_file, label_file in zip(image_files, label_files):
        print(f"Processing: {image_file.name}")
        image = reslice_to_90_degrees_view(io.imread(image_file))
        mask = reslice_to_90_degrees_view(io.imread(label_file))

        # Generate 5 augmentations for each image-mask pair
        for i in range(1, 6):
            suffix = f"_{i:03d}"  # Generate suffix with leading zeros
            output_path_img = output_dir_img / f"{image_file.stem}{suffix}.tiff"
            output_path_mask = output_dir_mask / f"{label_file.stem}{suffix}.tiff"
            augment_data(output_path_img, output_path_mask, image, mask, plot_result=False)




def augment_data(output_path_img, output_path_mask, image, mask, plot_result=False):
    # Convert numpy arrays to TorchIO images with proper data types
    image_tio = tio.ScalarImage(tensor=torch.tensor(image, dtype=torch.float32).unsqueeze(0))  # Add channel dimension
    mask_tio = tio.LabelMap(tensor=torch.tensor(mask, dtype=torch.int64).unsqueeze(0))        # Add channel dimension

    # Wrap the images in a TorchIO Subject
    subject = tio.Subject(
        image=image_tio,
        mask=mask_tio
    )

    # Define augmentation pipeline
    augmentation_pipeline = tio.Compose([
        tio.RandomAffine(scales=(0.9, 1.1), degrees=(0, 15, 15), isotropic=True),
        tio.RandomFlip(axes=(0, 1), flip_probability=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
        tio.RandomNoise(std=(0, 0.05), p=0.2),
        tio.RandomElasticDeformation(num_control_points=5, max_displacement=7, p=0.3),
        #tio.CropOrPad(target_shape=(96, 96, 64))
    ])

    # Apply augmentations to the subject
    augmented = augmentation_pipeline(subject)

    # Extract original and augmented images and masks
    orig_image = subject['image'].data.squeeze(0).numpy()  # Remove channel dimension
    orig_mask = subject['mask'].data.squeeze(0).numpy()
    aug_image = augmented['image'].data.squeeze(0).numpy()
    aug_mask = augmented['mask'].data.squeeze(0).numpy()

    if plot_result:
        plot_augmentation(orig_image, orig_mask, aug_image, aug_mask)

    resliced_img = reslice_to_90_degrees_view(aug_image)
    resliced_img = reslice_to_90_degrees_view(aug_image)
    resliced_img = reslice_to_90_degrees_view(aug_image)
    
    resliced_mask = reslice_to_90_degrees_view(aug_mask)
    resliced_mask = reslice_to_90_degrees_view(aug_mask)
    resliced_mask = reslice_to_90_degrees_view(aug_mask)

    save_tiff_stack(resliced_img, output_path_img)
    save_tiff_stack(resliced_mask, output_path_mask)


def plot_augmentation(orig_image, orig_mask, aug_image, aug_mask):
    # Visualization
    slice_index = 32  # Choose a middle slice for visualization
    plt.figure(figsize=(12, 12))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.title('Original Image (Slice)')
    plt.imshow(orig_image[:, :, slice_index], cmap='gray')
    plt.axis('off')

    # Original Mask
    plt.subplot(2, 2, 2)
    plt.title('Original Mask (Slice)')
    plt.imshow(orig_mask[:, :, slice_index], cmap='gray')
    plt.axis('off')

    # Augmented Image
    plt.subplot(2, 2, 3)
    plt.title('Augmented Image (Slice)')
    plt.imshow(aug_image[:, :, slice_index], cmap='gray')
    plt.axis('off')

    # Augmented Mask
    plt.subplot(2, 2, 4)
    plt.title('Augmented Mask (Slice)')
    plt.imshow(aug_mask[:, :, slice_index], cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()