import numpy as np
from scipy.ndimage import rotate, shift, gaussian_filter
from skimage.transform import rescale
import tifffile as tif
import os, sys
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))

src_dir = os.path.join(current_dir, '../..')
sys.path.append(src_dir)

def augment_3d(volume, labels=None, 
               rotation_range=(0, 360), 
               shift_range=(-5, 5), 
               scale_range=(0.8, 1.2), 
               intensity_variation=0.1, 
               add_noise=True, 
               sigma=1):
    """
    Perform 3D data augmentation on a volume and corresponding labels.
    See the previous explanation for details.
    """
    # Random rotation
    angle = np.random.uniform(*rotation_range)
    axes = [(0, 1), (0, 2), (1, 2)][np.random.choice(3)]  # Random axis pair
    volume = rotate(volume, angle=angle, axes=axes, reshape=False, mode='nearest')
    if labels is not None:
        labels = rotate(labels, angle=angle, axes=axes, reshape=False, mode='nearest', order=0)

    # Random shift (translation)
    shifts = np.random.uniform(*shift_range, size=3)  # (Z, Y, X)
    volume = shift(volume, shift=shifts, mode='nearest')
    if labels is not None:
        labels = shift(labels, shift=shifts, mode='nearest', order=0)

    # Random scaling
    scale_factor = np.random.uniform(*scale_range)
    volume = rescale(volume, scale_factor, mode='constant', anti_aliasing=True, preserve_range=True)
    if labels is not None:
        labels = rescale(labels, scale_factor, mode='constant', preserve_range=True, order=0)

    # Random intensity variation
    intensity_factor = 1 + np.random.uniform(-intensity_variation, intensity_variation)
    volume = volume * intensity_factor
    volume = np.clip(volume, 0, 1)  # Ensure values remain valid (for normalized images)

    # Add Gaussian noise
    if add_noise:
        noise = np.random.normal(0, sigma, size=volume.shape)
        volume = volume + noise
        volume = np.clip(volume, 0, 1)  # Ensure values remain valid

    # Gaussian smoothing
    volume = gaussian_filter(volume, sigma=sigma)

    # Return augmented data
    if labels is not None:
        return volume, labels
    else:
        return volume


def main():
    # Load the TIFF file
    current_directory = Path(__file__).resolve().parent.parent.parent
    input_image_dir = os.path.join(src_dir, 'ML_TrainingData/StarDist3D/Training_Images')
    input_label_dir = os.path.join(src_dir, 'ML_TrainingData/StarDist3D/Training_Masks')
    output_image_dir = current_directory / "output_images" / "augmented_images"
    output_label_dir = current_directory / "output_images" / "augmented_masks"

    # Create the output directory if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)


    # Number of augmentations per image
    num_augmentations = 5

    # Iterate over all files in the input directories
    for filename in os.listdir(input_image_dir):
        if filename.endswith(".tiff"):  # Process only TIFF files
            image_path = os.path.join(input_image_dir, filename)
            label_path = os.path.join(input_label_dir, filename)

            # Load the image and corresponding label
            volume = tif.imread(image_path)  # Shape: (Z, Y, X)
            labels = tif.imread(label_path)  # Shape: (Z, Y, X)

            # Perform augmentations
            for i in range(num_augmentations):
                # Augment the volume and labels
                augmented_volume, augmented_labels = augment_3d(volume, labels)

                # Save the augmented data
                augmented_image_path = os.path.join(output_image_dir, f"aug_{i}_{filename}")
                augmented_label_path = os.path.join(output_label_dir, f"aug_{i}_{filename}")

                tif.imwrite(augmented_image_path, augmented_volume.astype(np.float32))
                tif.imwrite(augmented_label_path, augmented_labels.astype(np.uint16))

            print(f"Augmented {filename} {num_augmentations} times.")

    print("All training images augmented.")
    
if __name__ == "__main__":
    main()