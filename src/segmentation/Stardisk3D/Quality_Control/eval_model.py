import os
import csv
import subprocess
import numpy as np
import random
from glob import glob
from astropy.visualization import simple_norm
from stardist import fill_label_holes, random_label_cmap
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
import shutil
from stardist.models import Config3D, StarDist3D
from fpdf import FPDF, HTMLMixin
from skimage import io
from skimage.io import imread
import pandas as pd
from datetime import datetime
from pip._internal.operations.freeze import freeze
from PIL import Image
# model name and path
#@markdown ###Do you want to assess the model you just trained ?
Use_the_current_trained_model = True #@param {type:"boolean"}

# #@markdown ###If not, please provide the path to the model folder:

# QC_model_folder = "" #@param {type:"string"}

# #Here we define the loaded model name and path
# QC_model_name = os.path.basename(QC_model_folder)
# QC_model_path = os.path.dirname(QC_model_folder)

class QualityControl:
    def __init__(self, td):
        self.QC_model_name = td.model_name
        self.QC_model_path = td.model_path
        self.QC_train = td.QC_source_images
        self.QC_target = td.QC_target_images
        self.full_QC_model_path = self.QC_model_path+'/'+self.QC_model_name+'/'

        self.pdResults = None
        
        if os.path.exists(self.full_QC_model_path):
            print("The "+self.QC_model_name+" network will be evaluated")
        else:
            W  = '\033[0m'  # white (normal)
            R  = '\033[31m' # red
            print(R+'!! WARNING: The chosen model does not exist !!'+W)
            print('Please make sure you provide a valid model path and model name before proceeding further.')

    #plot of training errors vs. epoch number
    def plt_training_error(self):
        lossDataFromCSV = []
        vallossDataFromCSV = []

        with open(self.QC_model_path+'/'+self.QC_model_name+'/Quality Control/training_evaluation.csv','r') as csvfile:
            csvRead = csv.reader(csvfile, delimiter=',')
            next(csvRead)
            for row in csvRead:
                lossDataFromCSV.append(float(row[0]))
                vallossDataFromCSV.append(float(row[1]))

        epochNumber = range(len(lossDataFromCSV))
        plt.figure(figsize=(15,10))

        plt.subplot(2,1,1)
        plt.plot(epochNumber,lossDataFromCSV, label='Training loss')
        plt.plot(epochNumber,vallossDataFromCSV, label='Validation loss')
        plt.title('Training loss and validation loss vs. epoch number (linear scale)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch number')
        plt.legend()


        plt.subplot(2,1,2)
        plt.semilogy(epochNumber,lossDataFromCSV, label='Training loss')
        plt.semilogy(epochNumber,vallossDataFromCSV, label='Validation loss')
        plt.title('Training loss and validation loss vs. epoch number (log scale)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch number')
        plt.legend()
        plt.savefig(self.QC_model_path+'/'+self.QC_model_name+'/Quality Control/lossCurvePlots.png',bbox_inches='tight',pad_inches=0)
        plt.show()



    def quality_metrics_estimation(self):
        Source_QC_folder = self.QC_train 
        Target_QC_folder = self.QC_target 

# #Here we allow the user to choose the number of tile to be used when predicting the images
# #@markdown #####To analyse large image, your images need to be divided into tiles.  Each tile will then be processed independently and re-assembled to generate the final image. "Automatic_number_of_tiles" will search for and use the smallest number of tiles that can be used, at the expanse of your runtime. Alternatively, manually input the number of tiles in each dimension to be used to process your images. 

# Automatic_number_of_tiles = False #@param {type:"boolean"}
# #@markdown #####If you get an Out of memory (OOM) error when using the "Automatic_number_of_tiles" option, disable it and manually input the values to be used to process your images.  Progressively increases these numbers until the OOM error disappear.
# n_tiles_Z =  1#@param {type:"number"}
# n_tiles_Y =  1#@param {type:"number"}
# n_tiles_X =  1#@param {type:"number"}

# if (Automatic_number_of_tiles): 
#   n_tilesZYX = None

# if not (Automatic_number_of_tiles):
#   n_tilesZYX = (n_tiles_Z, n_tiles_Y, n_tiles_X)

    
        #Create a quality control Folder and check if the folder already exist
        if os.path.exists(self.QC_model_path+"/"+self.QC_model_name+"/Quality Control") == False:
            os.makedirs(self.QC_model_path+"/"+self.QC_model_name+"/Quality Control")

        if os.path.exists(self.QC_model_path+"/"+self.QC_model_name+"/Quality Control/Prediction"):
            shutil.rmtree(self.QC_model_path+"/"+self.QC_model_name+"/Quality Control/Prediction")

        os.makedirs(self.QC_model_path+"/"+self.QC_model_name+"/Quality Control/Prediction")


        # Generate predictions from the Source_QC_folder and save them in the QC folder

        Source_QC_folder_tif = Source_QC_folder+"/*.tiff"


        np.random.seed(16)
        lbl_cmap = random_label_cmap()
        Z = sorted(glob(Source_QC_folder_tif))
        Z = list(map(imread,Z))
        n_channel = 1 if Z[0].ndim == 2 else Z[0].shape[-1]
        axis_norm = (0,1)   # normalize channels independently

        print('Number of test dataset found in the folder: '+str(len(Z)))

        
        # axis_norm = (0,1,2) # normalize channels jointly
        if n_channel > 1:
            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

        model = StarDist3D(None, name=self.QC_model_name, basedir=self.QC_model_path)

        names = [os.path.basename(f) for f in sorted(glob(Source_QC_folder_tif))]

        
        # modify the names to suitable form: path_images/image_numberX.tif
        
        lenght_of_Z = len(Z)
        
        for i in range(lenght_of_Z):
            img = normalize(Z[i], 1,99.8, axis=axis_norm)
            labels, polygons = model.predict_instances(img, prob_thresh=0.05, n_tiles=None)
            os.chdir(self.QC_model_path+"/"+self.QC_model_name+"/Quality Control/Prediction")
            #imsave(names[i], labels, polygons)
            save_tiff_imagej_compatible(names[i], labels, axes='ZYX')


        # Here we start testing the differences between GT and predicted masks


        with open(self.QC_model_path+"/"+self.QC_model_name+"/Quality Control/Quality_Control for "+self.QC_model_name+".csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["image","Prediction v. GT Intersection over Union"])  

            # Initialise the lists 
            filename_list = []
            IoU_score_list = []

            for n in os.listdir(Source_QC_folder):
                if not os.path.isdir(os.path.join(Source_QC_folder,n)):
                    print('Running QC on: '+n)
                    
                    test_input = io.imread(os.path.join(Source_QC_folder,n))
                    test_prediction = io.imread(os.path.join(self.QC_model_path+"/"+self.QC_model_name+"/Quality Control/Prediction",n))
                    test_ground_truth_image = io.imread(os.path.join(Target_QC_folder, n))

                #Convert pixel values to 0 or 255
                    test_prediction_0_to_255 = test_prediction
                    test_prediction_0_to_255[test_prediction_0_to_255>0] = 255

                #Convert pixel values to 0 or 255
                    test_ground_truth_0_to_255 = test_ground_truth_image
                    test_ground_truth_0_to_255[test_ground_truth_0_to_255>0] = 255

                # Intersection over Union metric

                    intersection = np.logical_and(test_ground_truth_0_to_255, test_prediction_0_to_255)
                    union = np.logical_or(test_ground_truth_0_to_255, test_prediction_0_to_255)
                    iou_score =  np.sum(intersection) / np.sum(union)
                    writer.writerow([n, str(iou_score)])

                    print("IoU: "+str(round(iou_score,3)))

                    filename_list.append(n)
                    IoU_score_list.append(iou_score)



        # Table with metrics as dataframe output
        self.pdResults = pd.DataFrame(index = filename_list)
        self.pdResults["IoU"] = IoU_score_list

        # Display results
        self.pdResults.head()


        # ------------- For display ------------
        print('--------------------------------------------------------------')
        self.show_QC_results()


    def show_QC_results(self):
        print("now comes the plot")

        # Select random file from QC folder
        Source_QC_folder = self.QC_train
        Target_QC_folder = self.QC_target

        src_files = [f for f in os.listdir(Source_QC_folder) if f.endswith('.tiff')]
        gt_files = [f for f in os.listdir(Target_QC_folder) if f.endswith('.tiff')]

        if not src_files or not gt_files:
            raise FileNotFoundError(f"Keine Dateien im Ordner {Source_QC_folder}, bzw {Target_QC_folder} gefunden.")

        random_src_file = random.choice(src_files)
        random_gt_file = random.choice(gt_files)

        src_path = os.path.join(Source_QC_folder, random_src_file)
        gt_path = os.path.join(Target_QC_folder, random_gt_file)

        # Load images
        test_input = io.imread(src_path)
        test_prediction = io.imread(os.path.join(self.QC_model_path, self.QC_model_name, "Quality Control/Prediction", random_src_file))
        test_ground_truth_image = io.imread(gt_path)

        # Convert pixel values to 0 or 255
        test_prediction_0_to_255 = (test_prediction > 0).astype(np.uint8) * 255
        test_ground_truth_0_to_255 = (test_ground_truth_image > 0).astype(np.uint8) * 255

        # Get image dimensions
        Image_Z = test_input.shape[0]

        slice_idx = 36  # Start with slice ..

        # Normalize input image
        norm = mcolors.Normalize(vmin=np.percentile(test_input, 1), vmax=np.percentile(test_input, 99))

        # Set up figure and axes
        fig, axes = plt.subplots(1, 4, figsize=(32, 8))

        # Initialize plots (showing slice 25 initially)
        im_input = axes[0].imshow(test_input[slice_idx], norm=norm, cmap='magma', interpolation='nearest')
        
        # Overlay (Input Image and Prediction Mask) in the second axes
        im_overlay_input = axes[1].imshow(test_input[slice_idx], cmap='magma', interpolation='nearest')  # Full opacity for input
        im_overlay_pred = axes[1].imshow(test_prediction[slice_idx], alpha=0.5, cmap='Blues')  # Blue prediction mask (with transparency)

        # Display the prediction image in the third axis (full opacity)
        im_pred = axes[2].imshow(test_prediction[slice_idx], cmap='Blues', interpolation='nearest')

        # Ground truth in the last axes
        im_gt = axes[3].imshow(test_ground_truth_image[slice_idx], interpolation='nearest', cmap='Greens')

        # Titles
        axes[0].set_title("Input")
        # axes[1].set_title("Overlay")
        # axes[2].set_title("Prediction")
        axes[3].set_title(f"Ground Truth - IoU: {round(self.pdResults.loc[random_src_file]['IoU'], 3)}")

        for ax in axes:
            ax.axis("off")

        # Add single slider for slice selection
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.02])
        slider = Slider(ax_slider, "Slice", 0, Image_Z - 1, valinit=slice_idx, valstep=1)


        # Function to update plots when slider moves
        def update(val):
            slice_idx = int(slider.val)

            # Update input image
            im_input.set_data(test_input[slice_idx])

            # Update overlay images (input stays the same, prediction updates)
            im_overlay_input.set_data(test_input[slice_idx])  # Input image
            im_overlay_pred.set_data(test_prediction_0_to_255[slice_idx])  # Prediction mask

            # Update prediction and ground truth images
            im_pred.set_data(test_prediction_0_to_255[slice_idx])  
            im_gt.set_data(test_ground_truth_0_to_255[slice_idx])  

            # Redraw figure
            fig.canvas.draw_idle()

        # Attach slider to the update function
        slider.on_changed(update)

        plt.show()

