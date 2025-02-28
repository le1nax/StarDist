import os, sys

# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__)) 

# Append the 'plotting' directory to sys.path
src_dir = os.path.join(current_dir, '../..')
sys.path.append(src_dir)


from StardiskModelCreation import StarDiskModel
from StarDiskTraining import TrainStarDisk
from input_config import InputConfig as cfg
from TrainingParameters import TrainingData
from stardisk_utils import check_gpu 
from Quality_Control.eval_model import QualityControl
from model_application.apply_model import StarDiskApplication

# Display TensorFlow version
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

#warnings.filterwarnings("ignore")

def initTrainingParametersFromConfig():
    td = TrainingData(Training_source=cfg.Training_source,
                      Training_target=cfg.Training_target,
                      QC_source_images=cfg.QC_source_images,
                      QC_target_images=cfg.QC_target_images,
                      model_path=cfg.model_path,
                      number_of_epochs=cfg.number_of_epochs,
                      Use_Data_augmentation=cfg.Use_Data_augmentation,
                      Use_Default_Advanced_Parameters=cfg.Use_Default_Advanced_Parameters,
                      batch_size=cfg.batch_size,
                      number_of_steps=cfg.number_of_steps,
                      patch_size=cfg.patch_size,
                      patch_height=cfg.patch_height,
                      percentage_validation=cfg.percentage_validation,
                      n_rays=cfg.n_rays,         
                      initial_learning_rate=cfg.initial_learning_rate)
    return td
# Main script
if __name__ == "__main__":
    #TRAINING PIPEPLINE

    #INIT CLASSES
    print("Running StarDist model script")
    check_gpu()
    td = initTrainingParametersFromConfig()
    # model = StarDiskModel(td)
    # trainer = TrainStarDisk(model)
    
    # #TRAIN MODEL
    # trainer.train_model()

    # #EVALUATE TRAINING
    # qc = QualityControl(td)
    # qc.plt_training_error()
    # qc.quality_metrics_estimation()
    # qc.show_QC_results()

    # USE MODEL
    a = StarDiskApplication()
    a.predict_data(show_patches=True)
    a.show_QC_results(save_figure=True)


    print("Script completed")

