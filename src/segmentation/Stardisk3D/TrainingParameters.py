from __future__ import print_function, unicode_literals, absolute_import, division
import os
import pandas as pd
import sys
import warnings
import numpy as np
from builtins import any as b_any
from stardist import fill_label_holes, random_label_cmap
from stardist.models import Config3D, StarDist3D
from stardist import relabel_image_stardist3D, Rays_GoldenSpiral
from csbdeep.utils import Path, normalize
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from IPython.display import display
from tifffile import imread, imsave
from skimage.util import img_as_uint
from skimage import img_as_float32, io
from fpdf import FPDF, HTMLMixin
from datetime import datetime
from pip._internal.operations.freeze import freeze
from astropy.visualization import simple_norm
import subprocess
import csv


warnings.filterwarnings("ignore")

import random

import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(project_dir)

# Colors for the warning messages
class bcolors:
    WARNING = '\033[31m'
    W  = '\033[0m'  # white (normal)
    R  = '\033[31m' # red


#@todo add option for pretrained model, Use_Default_Advanced_Parameters
class TrainingData:
    def __init__(self, Training_source, Training_target,
                 QC_source_images, QC_target_images, model_path,
                 model_name = 'Stardisk3D_model',
                 number_of_epochs=100,
                 Use_Data_augmentation=True,
                 Use_Default_Advanced_Parameters=True,
                 batch_size=2,
                 number_of_steps=0,
                 patch_size=96,
                 patch_height=48,
                 percentage_validation=10,
                 n_rays=96,
                 initial_learning_rate=0.0003
                 ):
       # Initialize all member variables with input arguments
        self.Training_source = Training_source
        self.Training_target = Training_target
        self.QC_source_images = QC_source_images
        self.QC_target_images = QC_target_images
        self.model_path = model_path
        self.model_name = model_name
        self.trained_model = model_path
        self.number_of_epochs = number_of_epochs
        self.Use_Default_Advanced_Parameters = Use_Default_Advanced_Parameters
        self.Use_Data_augmentation = Use_Data_augmentation
        self.batch_size = batch_size
        self.number_of_steps = number_of_steps
        self.patch_size = patch_size
        self.patch_height = patch_height
        self.percentage_validation = percentage_validation
        self.n_rays = n_rays
        self.initial_learning_rate = initial_learning_rate
        self.Image_X = None
        self.Image_Y = None
        self.Image_Z = None
        self.mid_plane = None

        # Derived or computed member variables
        self.percentage = percentage_validation / 100  # Convert percentage to a fraction
        
        self.load_all_images()
        
        self.verify_input_shape()
        self.verify_patch_size()
        self.visualize_training_data()

    def load_all_images(self):
        source_files = [f for f in os.listdir(self.Training_source) if f.endswith('.tiff')]
        target_files = [f for f in os.listdir(self.Training_target) if f.endswith('.tiff')]

        if len(source_files) != len(target_files):
            raise ValueError("The number of source and target files does not match.")

        # Initialize arrays to hold the images
        images_list = []

        # Load all the images from the source and target directories
        for src_file, tgt_file in zip(source_files, target_files):
            src_image = imread(os.path.join(self.Training_source, src_file))
            tgt_image = imread(os.path.join(self.Training_target, tgt_file))
            
            images_list.append((src_image, tgt_image))

        # Convert list to a 4D numpy array (images, z, y, x)
        self.x = np.array([item[0] for item in images_list])  # Source images
        self.y = np.array([item[1] for item in images_list])  # Target images

        print(f"Loaded {self.x.shape[0]} source images and {self.y.shape[0]} target images.")

    def verify_input_shape(self):
        # Iterate over each image in self.x and self.y
        for idx, (img_x, img_y) in enumerate(zip(self.x, self.y)):
            # Check if self.x image shape is (64, 128, 128)
            if img_x.shape != (64, 128, 128):
                warnings.warn(f"Image {idx+1} in self.x does not have the correct shape. Expected (64, 128, 128), found {img_x.shape}")
            
            # Check if self.y image shape is (64, 128, 128)
            if img_y.shape != (64, 128, 128):
                warnings.warn(f"Image {idx+1} in self.y does not have the correct shape. Expected (64, 128, 128), found {img_y.shape}")
            
    def verify_patch_size(self):
        # Iterate over each image in self.x and self.y
        for idx, (img_x, img_y) in enumerate(zip(self.x, self.y)):
            # Get image dimensions
            self.Image_Z = img_x.shape[0]
            self.Image_Y = img_x.shape[1]
            self.Image_X = img_x.shape[2]

            # Select the mid-plane
            mid_plane = int(self.Image_Z / 2) + 1

            # Check if patch_size is smaller than the smallest xy dimension of the image
            if self.patch_size > min(self.Image_Y, self.Image_X):
                self.patch_size = min(self.Image_Y, self.Image_X)
                print(f"Warning: Image {idx+1} - Your chosen patch_size is bigger than the xy dimension of your image; patch_size is now: {self.patch_size}")

            # Check if patch_size is divisible by 8
            if not self.patch_size % 8 == 0:
                self.patch_size = ((int(self.patch_size / 8)-1) * 8)
                print(f"Warning: Image {idx+1} - Your chosen patch_size is not divisible by 8; patch_size is now: {self.patch_size}")

            # Check if patch_height is smaller than the z dimension of the image
            if self.patch_height > self.Image_Z:
                self.patch_height = self.Image_Z
                print(f"Warning: Image {idx+1} - Your chosen patch_height is bigger than the z dimension of your image; patch_height is now: {self.patch_height}")

            # Check if patch_height is divisible by 4
            if not self.patch_height % 4 == 0:
                self.patch_height = ((int(self.patch_height / 4)-1) * 4)
            if self.patch_height == 0:
                self.patch_height = 4
                print(f"Warning: Image {idx+1} - Your chosen patch_height is not divisible by 4; patch_height is now: {self.patch_height}")

    def visualize_training_data(self):
        os.chdir(self.Training_target)
        # Normalize input image
        norm = mcolors.Normalize(vmin=np.percentile(self.x, 1), vmax=np.percentile(self.x, 99))

        # Get image dimensions
        num_images = self.x.shape[0]  # N (number of images)
        Image_Z = self.x.shape[1]  # Z (slices per image)
        Image_Y = self.x.shape[2]  # Y (height)
        Image_X = self.x.shape[3]  # X (width)

        # Set up figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Initially display the first image (index 0) and the slice 25
        image_idx = 0  # Start with the first image (index 0)
        slice_idx = 25  # Start with slice 25
        print("DEBUG TRAIN DATA TYPE")
        print(self.x[0].dtype)
        print(self.x[0].max)
        im_input = axes[0].imshow(self.x[image_idx, slice_idx], interpolation='nearest', norm=norm, cmap='magma')
        axes[0].axis('off')
        axes[0].set_title(f'Training source (Image={image_idx}, Z={slice_idx})')

        im_target = axes[1].imshow(self.y[image_idx, slice_idx], interpolation='nearest', cmap='Greens')
        axes[1].axis('off')
        axes[1].set_title(f'Training target (Image={image_idx}, Z={slice_idx})')

        # Add a slider for slice selection
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.02])  # Positioning of slider
        slider = Slider(ax_slider, "Slice", 0, Image_Z - 1, valinit=slice_idx, valstep=1)

        # Function to update the images when the slider value changes
        def update(val):
            slice_idx = int(slider.val)  # Get the current slice index
            im_input.set_data(self.x[image_idx, slice_idx])  # Update input image
            im_target.set_data(self.y[image_idx, slice_idx])  # Update target image
            fig.canvas.draw_idle()  # Redraw the figure

        # Attach the slider to the update function
        slider.on_changed(update)

        # Function to handle text input for both image and slice index
        def on_text_submit(text, is_image=True):
            nonlocal image_idx, slice_idx
            try:
                if is_image:
                    # Image selection via TextBox
                    image_idx = int(text)
                    if image_idx < 0 or image_idx >= num_images:
                        print(f"Invalid image index: {image_idx}. Please enter a value between 0 and {num_images - 1}.")
                        return
                    # Update the image display
                    im_input.set_data(self.x[image_idx, slice_idx])  # Update input image
                    im_target.set_data(self.y[image_idx, slice_idx])  # Update target image
                    axes[0].set_title(f'Training source (Image={image_idx}, Z={slice_idx})')
                    axes[1].set_title(f'Training target (Image={image_idx}, Z={slice_idx})')
                else:
                    # Slice selection via TextBox
                    slice_idx = int(text)
                    if slice_idx < 0 or slice_idx >= Image_Z:
                        print(f"Invalid slice index: {slice_idx}. Please enter a value between 0 and {Image_Z - 1}.")
                        return
                    # Update the slice display
                    slider.set_val(slice_idx)  # Update slider
                    im_input.set_data(self.x[image_idx, slice_idx])  # Update input image
                    im_target.set_data(self.y[image_idx, slice_idx])  # Update target image
                    axes[0].set_title(f'Training source (Image={image_idx}, Z={slice_idx})')
                    axes[1].set_title(f'Training target (Image={image_idx}, Z={slice_idx})')
            except ValueError:
                print("Please enter a valid integer.")

        # Create text boxes for both image and slice index selection
        ax_image_textbox = plt.axes([0.01, 0.02, 0.15, 0.05])  # Positioning of the image textbox
        text_box_image = TextBox(ax_image_textbox, "Image Index:", initial="0")
        text_box_image.on_submit(lambda text: on_text_submit(text, is_image=True))

        ax_slice_textbox = plt.axes([0.2, 0.02, 0.15, 0.05])  # Positioning of the slice textbox
        text_box_slice = TextBox(ax_slice_textbox, "Slice Index:", initial="25")
        text_box_slice.on_submit(lambda text: on_text_submit(text, is_image=False))

        #Show the plot
        plt.show()

    def augmenter(self, x, y):
        # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
        # as 3D microscopy acquisitions are usually not axially symmetric
        x, y = self.random_fliprot(x, y, axis=(1,2))
        x = self.random_intensity_change(x)
        return x, y
    
    def random_intensity_change(self, img):
        img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
        return img


    def random_fliprot(self, img, mask, axis=None): 
        if axis is None:
            axis = tuple(range(mask.ndim))
        axis = tuple(axis)
                
        assert img.ndim>=mask.ndim
        perm = tuple(np.random.permutation(axis))
        transpose_axis = np.arange(mask.ndim)
        for a, p in zip(axis, perm):
            transpose_axis[a] = p
        transpose_axis = tuple(transpose_axis)
        img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
        mask = mask.transpose(transpose_axis) 
        for ax in axis: 
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=ax)
                mask = np.flip(mask, axis=ax)
        return img, mask 
        

