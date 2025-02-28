from __future__ import print_function, unicode_literals, absolute_import, division
import os
import pandas as pd
import sys
import warnings
import numpy as np
from builtins import any as b_any
from stardist import fill_label_holes, random_label_cmap
from stardist.models import Config3D, StarDist3D, StarDistData3D
from stardist import relabel_image_stardist3D, Rays_GoldenSpiral
from csbdeep.utils import Path, normalize
from matplotlib import pyplot as plt
import numpy as np
from tifffile import imread, imsave
from skimage.util import img_as_uint
from skimage import img_as_float32, io
from fpdf import FPDF, HTMLMixin
from datetime import datetime
from pip._internal.operations.freeze import freeze
from astropy.visualization import simple_norm
from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import relabel_image_stardist3D, Rays_GoldenSpiral, calculate_extents
from stardist.matching import matching_dataset
import subprocess
import shutil
from tqdm import tqdm 
import csv
from glob import glob


warnings.filterwarnings("ignore")

import random

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_dir, '..')
sys.path.append(project_dir)
from TrainingParameters import TrainingData, bcolors

class StarDiskModel:
    def __init__(self, training_data):
        self.td = training_data

        # import training images and masks and sorts them suitable for the network
        training_images_tiff = self.td.Training_source+"/*.tiff" #note that at fraunhofer, images are saved as .tiff with two f
        mask_images_tiff = self.td.Training_target+"/*.tiff"
        self.X = sorted(glob(training_images_tiff))   
        self.Y = sorted(glob(mask_images_tiff))
        self.n_channel = 1
        self.Use_Data_augmentation = self.td.Use_Data_augmentation
        self.use_gpu = True
        self.Use_pretrained_model = False
        self.X_val = None
        self.Y_val = None
        self.X_trn = None
        self.Y_trn = None
        self.conf = None
        self.build_model()
        

    def build_model(self):
        self.verify_unique_model()
        self.verify_images()
        self.ensure_min_nsteps()
        self.map_norm_XY()
        self.split_train_val_data()
        self.ensure_min_nsteps()
        self.use_pretrained_model()
        self.build_config()
        self.create_model()
        self.verify_fov()
        

    def verify_unique_model(self):
        if os.path.exists(self.td.model_path+'/'+self.td.model_name):
            print(bcolors.WARNING +"!! WARNING: Model folder already exists and has been removed !!")
            shutil.rmtree(self.td.model_path+'/'+self.td.model_name)
    
    def verify_images(self):
        # assert -funtion check that X and Y really have images. If not this cell raises an error
        assert all(Path(x).name==Path(y).name for x,y in zip(self.X,self.Y))

    def map_norm_XY(self):
       # Here we map the training dataset (images and masks).
        self.X = list(map(imread,self.X))
        self.Y = list(map(imread,self.Y))

        n_channel = 1 if self.X[0].ndim == 3 else self.X[0].shape[-1]



        #Normalize images and fill small label holes.
        axis_norm = (0,1,2)   # normalize channels independently
        # axis_norm = (0,1,2,3) # normalize channels jointly
        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
            sys.stdout.flush()

        self.X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(self.X)]
        self.Y = [fill_label_holes(y) for y in tqdm(self.Y)]

    def split_train_val_data(self):

        assert len(self.X) > 1, "not enough training data"
        rng = np.random.RandomState(42)
        ind = rng.permutation(len(self.X))
        n_val = max(1, int(round(self.td.percentage * len(ind))))
        ind_train, ind_val = ind[:-n_val], ind[-n_val:]
        self.X_val, self.Y_val = [self.X[i] for i in ind_val]  , [self.Y[i] for i in ind_val]
        self.X_trn, self.Y_trn = [self.X[i] for i in ind_train], [self.Y[i] for i in ind_train] 
        print('number of images: %3d' % len(self.X))
        print('- training:       %3d' % len(self.X_trn))
        print('- validation:     %3d' % len(self.X_val))


        extents = calculate_extents(self.Y)
        self.anisotropy = tuple(np.max(extents) / extents)
        print('empirical anisotropy of labeled objects = %s' % str(self.anisotropy))

    def ensure_min_nsteps(self):
       # Use OpenCL-based computations for data generator during training (requires 'gputools')
        self.use_gpu = False and gputools_available()

        #Here we ensure that our network has a minimal number of steps
        if (self.td.Use_Default_Advanced_Parameters) or (self.td.number_of_steps == 0):
            self.td.number_of_steps = (self.td.Image_X//self.td.patch_size)*(self.td.Image_Y//self.td.patch_size)*(self.td.Image_Z//self.td.patch_height) * int(len(self.X)/self.td.batch_size)+1
        if (self.td.Use_Data_augmentation):
            self.td.number_of_steps = self.td.number_of_steps*6

        print("Number of steps: "+str(self.td.number_of_steps))


    def use_pretrained_model(self):
        
        # --------------------- Using pretrained model ------------------------
        #Here we ensure that the learning rate set correctly when using pre-trained models
        # if self.Use_pretrained_model:
        #     if self.Weights_choice == "last":
        #         initial_learning_rate = lastLearningRate

        #     if selfWeights_choice == "best":            
        #         initial_learning_rate = bestLearningRate
        # --------------------- ---------------------- ------------------------
        return #@todo

    def build_config(self):

        # Predict on subsampled grid for increased efficiency and larger field of view
        grid = tuple(1 if a > 1.5 else 2 for a in self.anisotropy)

        # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
        rays = Rays_GoldenSpiral(self.td.n_rays, anisotropy=self.anisotropy)

        self.conf = Config3D (
            rays             = rays,
            grid             = grid,
            anisotropy       = self.anisotropy,
            use_gpu          = self.use_gpu,
            n_channel_in     = self.n_channel,
            train_learning_rate = self.td.initial_learning_rate,
            train_patch_size = (self.td.patch_height, self.td.patch_size,  self.td.patch_size),
            train_batch_size = self.td.batch_size,
        )
        print(self.conf)
        vars(self.conf)



# --------------------- This is currently disabled as it give an error ------------------------
# #here we limit GPU to 80%
# if use_gpu:
#     from csbdeep.utils.tf import limit_gpu_memory
#     # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
#     limit_gpu_memory(0.8)
# # --------------------- ---------------------- ------------------------

    def create_model(self):
        # Here we create a model according to section 5.3.
        self.model = StarDist3D(self.conf, name=self.td.model_name, basedir=self.td.trained_model)

# @todo # --------------------- Using pretrained model ------------------------
# # Load the pretrained weights 
# if Use_pretrained_model:
#   model.load_weights(h5_file_path)
# # --------------------- ---------------------- ------------------------

    def verify_fov(self):
        #Here we check the FOV of the network.
        median_size = calculate_extents(self.Y, np.median)
        fov = np.array(self.model._axes_tile_overlap('ZYX'))
        if any(median_size > fov):
            print("WARNING: median object size larger than field of view of the neural network.")
            
