from PIL import Image
import numpy as np
import os, sys
from stardist import fill_label_holes, random_label_cmap
from glob import glob
from skimage.io import imread
from stardist.models import StarDist3D
from csbdeep.utils import normalize
from csbdeep.io import save_tiff_imagej_compatible
import matplotlib.pyplot as plt
from skimage import io
import random
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from skimage import filters


import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist3D
from skimage.io import imread
import os
from scipy.ndimage import zoom

import imageio
import os

# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the 'plotting' directory to sys.path
src_dir = os.path.join(current_dir, '../../../..')
sys.path.append(src_dir)


from input_config import InputConfig as cfg
from cropping.cropSlice_toTrainingData import resize_to_multiple, crop_image_to_patches, stitch_patches_back
from plotting.masking import show_masks
from fileLoadingSaving.loadSaveTiff import save_tiff_stack

class StarDiskApplication:
    def __init__(self):
        self.Data_folder = cfg.QC_source_images
        src_dir = os.path.join(current_dir, '../../../..')
        sys.path.append(src_dir)
        src_dir = Path(src_dir)
        self.Results_folder = src_dir / "output_images" / "segmented_images"
        self.Prediction_model_name = cfg.model_name
        self.Prediction_model_path = cfg.model_path

        self.axis_norm = None
        self.lbl_cmap = None   # normalize channels independently
        #Here we allow the user to choose the number of tile to be used when predicting the images
        #@markdown #####To analyse large image, your images need to be divided into tiles.  Each tile will then be processed independently and re-assembled to generate the final image. "Automatic_number_of_tiles" will search for and use the smallest number of tiles that can be used, at the expanse of your runtime. Alternatively, manually input the number of tiles in each dimension to be used to process your images. 

        # Automatic_number_of_tiles = False #@param {type:"boolean"}
        # #@markdown #####If you get an Out of memory (OOM) error when using the "Automatic_number_of_tiles" option, disable it and manually input the values to be used to process your images.  Progressively increases these numbers until the OOM error disappear.
        # n_tiles_Z =  1#@param {type:"number"}
        # n_tiles_Y =  1#@param {type:"number"}
        # n_tiles_X =  1#@param {type:"number"}

        # if (Automatic_number_of_tiles): 
        #   n_tilesZYX = None

        # if not (Automatic_number_of_tiles):
        #   n_tilesZYX = (n_tiles_Z, n_tiles_Y, n_tiles_X)


        self.full_Prediction_model_path = self.Prediction_model_path+'/'
        if os.path.exists(self.full_Prediction_model_path):
            print("The "+ self.Prediction_model_name +" network will be used.")
        else:
            W  = '\033[0m'  # white (normal)
            R  = '\033[31m' # red
            print(R+'!! WARNING: The chosen model does not exist !!'+W)

    def process_and_predict_large_image(self, image, model, patch_size=(128, 256, 256), axis_norm=(0,1), show_patches=False, save_stacks=False):
        """
        Process a large 3D image with StarDist by resizing, cropping into patches, 
        predicting each patch, and stitching the results back.
        """
        original_shape = image.shape
        image = normalize(image, 1, 99.8, axis=axis_norm)
        image_resized, new_shape, zoom_factors = resize_to_multiple(image, patch_size)
        patches, patch_coords = crop_image_to_patches(image_resized, patch_size)
        
        patch_masks = []
        # Ensure output directories exist
        current_directory = Path(__file__).resolve().parent.parent.parent.parent.parent
        patch_output_dir = current_directory / "output_images" / "extracted_patches" 
        mask_output_dir = current_directory / "output_images" / "extracted_masks" 
        os.makedirs(patch_output_dir, exist_ok=True)
        os.makedirs(mask_output_dir, exist_ok=True)

        for i, patch in enumerate(patches):
            patch = filters.gaussian(patch, sigma=1) 
            labels, _ = model.predict_instances(patch, prob_thresh=0.01)
            
            if show_patches: 
                show_masks(patch, labels)
            
            patch_masks.append(labels)
            
            if(save_stacks):
                # Save patch and corresponding mask as TIFF
                patch_filename = os.path.join(patch_output_dir, f"patch_{i}.tiff")
                mask_filename = os.path.join(mask_output_dir, f"patch_{i}.tiff")
                
                save_tiff_stack(patch, patch_filename)
                save_tiff_stack(labels, mask_filename)

        full_mask = stitch_patches_back(patch_masks, patch_coords, new_shape, patch_size)
        
        # Resize the full mask back to the original input shape
        full_mask_resized = zoom(full_mask, (1/zoom_factors[0], 1/zoom_factors[1], 1/zoom_factors[2]), order=0)
        # mask_filename = os.path.join(mask_output_dir, f"application_output.tiff")
        # save_tiff_stack(full_mask_resized, mask_filename)
        
        return full_mask_resized



    def predict_data(self, thresh=0.01, show_patches=False):
        #single images
        #testDATA = test_dataset
        self.Dataset = self.Data_folder+"/*.tiff"


        np.random.seed(16)
        lbl_cmap = random_label_cmap()
        X = sorted(glob(self.Dataset))
        X = list(map(imread,X))
        n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
        
        self.axis_norm = (0,1)
        # axis_norm = (0,1,2) # normalize channels jointly
        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if self.axis_norm is None or 2 in self.axis_norm else 'independently'))
        model = StarDist3D(None, name=self.Prediction_model_name, basedir=self.Prediction_model_path)
        
        #Sorting and mapping original test dataset
        #X = sorted(glob(Dataset))
        #X = list(map(imread,X))
        names = [os.path.basename(f) for f in sorted(glob(self.Dataset))]

        # modify the names to suitable form: path_images/image_numberX.tif
        FILEnames=[]
        for m in names:
            m = self.Results_folder / m  
            FILEnames.append(str(m))  

        # Predictions folder
        X = sorted(glob(self.Dataset))
        X = list(map(imread,X))
        lenght_of_X = len(X)
        for i in range(lenght_of_X):
            img = normalize(X[i], 1,99.8, axis=self.axis_norm)
        
            labels = self.process_and_predict_large_image(img, model, show_patches=show_patches)
            # Save the predicted mask in the result folder
            os.chdir(self.Results_folder)
            #imsave(FILEnames[i], labels, polygons)
            save_tiff_imagej_compatible(FILEnames[i], labels, axes='ZYX')
        



        print("The mid-plane image is displayed below.")
        # ------------- For display ------------
        print('--------------------------------------------------------------')

    def show_QC_results(self, save_figure=False):

  
        print("now comes the plot")

       # pick random file from qc folder
        src_files = [f for f in os.listdir(self.Data_folder) if f.endswith('.tiff')]
        gt_files = [f for f in os.listdir(self.Results_folder) if f.endswith('.tiff')]

        random_src_file = random.choice(src_files)
        random_gt_file = random.choice(gt_files)

        src_path = os.path.join(self.Data_folder, random_src_file)
        gt_path = os.path.join(self.Results_folder, random_gt_file)

        test_input = io.imread(src_path)
        
        labels = io.imread(gt_path)



        
        show_masks(test_input, labels, save_figure=save_figure)

