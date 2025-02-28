import os, sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(project_dir)


class InputConfig:
  
    #Path to training images:
    Training_source = os.path.join(project_dir, 'ML_TrainingData/StarDist3D/Training_Images')
    Training_target = os.path.join(project_dir, 'ML_TrainingData/StarDist3D/Training_Masks')
    QC_source_images = os.path.join(project_dir, 'ML_TrainingData/StarDist3D/Quality Control/images')
    QC_target_images = os.path.join(project_dir, 'ML_TrainingData/StarDist3D/Quality Control/masks')

    #Name of the model and path to model folder:
    model_name = "default" 
    model_path = os.path.join(project_dir, 'output_data/Stardisk3D_model')

    # Other parameters for traini ng:
    number_of_epochs =  300

    Use_Data_augmentation = True
    #Advanced Parameters
    Use_Default_Advanced_Parameters = False

    #If not, please input:

    #GPU_limit = 90 
    batch_size =  2
    number_of_steps = 0
    patch_size =  96
    patch_height =  48
    percentage_validation =   10
    n_rays = 96
    initial_learning_rate = 0.0003

    if (Use_Default_Advanced_Parameters): 
        print("Default advanced parameters enabled")
        batch_size = 2 
        n_rays = 96
        percentage_validation = 10
        initial_learning_rate = 0.0003

    patch_size =  96
    patch_height =  48


    percentage = percentage_validation/100