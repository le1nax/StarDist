import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path

# def plotTiff2D(oct_data):
    
#     print(f"Data shape: {oct_data.shape}")

#     plt.imshow(oct_data, cmap='gray')
#     plt.title("2D Image")
#     plt.axis("off")  # Hide axes for better visualization
#     plt.show()
#plt.switch_backend('TkAgg')  # or 'Qt5Agg'
def plotTiff3D(image_data, is_list=False, ):


    # If not a list, proceed with single image plotting as before
    if not is_list:
        if image_data.ndim == 2:
            print(f"Data shape: {image_data.shape}")
            plt.imshow(image_data, cmap='gray')
            plt.title("2D Image")
            plt.axis("off")  # Hide axes for better visualization
            plt.show()
            print("")
            return
        elif image_data.ndim == 3:
            print(f"Data shape: {image_data.shape}")

            # If the 3rd dimension has value 3, treat it as RGB
            if image_data.shape[2] == 3:
                plt.imshow(image_data)
                plt.title("2D RGB Image")
                plt.axis("off")  # Hide axes for better visualization
                plt.show()
                print("")
                return

            # If it's a 3D volume, proceed to plot it
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Define 3D grid
            x, y, z = np.indices(image_data.shape)

            # Plot the 3D volume by projecting the intensity values onto the grid
            ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=image_data.flatten(), cmap='gray', marker='.')

            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            plt.show()
            return

    # If input is a list of images, proceed with new functionality
    image_list = image_data if is_list else [image_data]
    num_images = len(image_list)

    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 6, figsize=(15, 5 * num_images))

    # If only one image is in the list, adjust axes to handle single subplot case
    if num_images == 1:
        axes = [axes]

    for i, img in enumerate(image_list):

        # Flatten the axes array for simpler indexing
        axes = axes.flatten()
        ax = axes[i]
        # if not isinstance(img, np.ndarray):
        #     raise ValueError(f"Expected a NumPy array, but got {type(image_data)} instead.")
        
        # Check if the image is 2D or 3D and display accordingly
        if img.ndim == 2:
            if(i<6):
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Image {i+1}")
                ax.axis("off")
            if(i>5 and i<12):
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Image {i-5}")
                ax.axis("off")
            if(i>11):
                ax.imshow(img, cmap='gray')
                ax.set_title(f"Image {i-11}")
                ax.axis("off")
        elif img.ndim == 3:
            # Display as RGB if third dimension is 3, otherwise show a single slice
            if img.shape[2] == 3:
                ax.imshow(img)
                ax.set_title(f"Image {i+1} (RGB)")
                ax.axis("off")
            else:
                ax.imshow(img[:, :, img.shape[2] // 2], cmap='gray')  # Show a middle slice
                ax.set_title(f"Image {i+1} (Slice)")
                ax.axis("off")

    # Hide any unused subplots
    for j in range(len(image_list), len(axes)):
        axes[j].axis("off")    

    plt.tight_layout()
    plt.show()