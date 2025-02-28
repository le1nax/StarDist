import os, sys
import pandas as pd

import shutil

# Get the current directory (the directory of main_script.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the 'plotting' directory to sys.path
src_dir = os.path.join(current_dir, '../..')
sys.path.append(src_dir)

from StardiskModelCreation import StarDiskModel

import time
import csv

import warnings
warnings.filterwarnings("ignore")

#@markdown ##Start training

# augmenter = None

# def augmenter(X_batch, Y_batch):
#     """Augmentation for data batch.
#     X_batch is a list of input images (length at most batch_size)
#     Y_batch is the corresponding list of ground-truth label images
#     """
#     # ...
#     return X_batch, Y_batch

# Training the model. 
# 'input_epochs' and 'steps' refers to your input data in section 5.1 

class TrainStarDisk:
    def __init__(self, StarDiskModel):
        self.sd = StarDiskModel
        self.startTime = time.time()
        self.history = None
        self.lossData = None

    def train_model(self):
        self.history = self.sd.model.train(self.sd.X_trn, self.sd.Y_trn, validation_data=(self.sd.X_val,self.sd.Y_val), augmenter=self.sd.td.augmenter,
                            epochs=self.sd.td.number_of_epochs, steps_per_epoch=self.sd.td.number_of_steps)
        print("Training done")
        self.doc_training()
        self.optimize_network()
        self.export_training_summary()
    
    def doc_training(self):
        # convert the history.history dict to a pandas DataFrame:     
        self.lossData = pd.DataFrame(self.history.history) 

        if os.path.exists(self.sd.td.model_path+"/"+self.sd.td.model_name+"/Quality Control"):
            shutil.rmtree(self.sd.td.model_path+"/"+self.sd.td.model_name+"/Quality Control")

        os.makedirs(self.sd.td.model_path+"/"+self.sd.td.model_name+"/Quality Control")

        # The training evaluation.csv is saved (overwrites the Files if needed). 
        lossDataCSVpath = self.sd.td.model_path+'/'+self.sd.td.model_name+'/Quality Control/training_evaluation.csv'
        with open(lossDataCSVpath, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['loss','val_loss'])#, 'learning rate'])
            for i in range(len(self.history.history['loss'])):
                #writer.writerow([self.history.history['loss'][i], self.history.history['val_loss'][i], self.history.history['lr'][i]])
                writer.writerow([self.history.history['loss'][i], self.history.history['val_loss'][i]])
                print("loss documented")

    def optimize_network(self):
        print("Network optimization in progress")

        #Here we optimize the network.
        self.sd.model.optimize_thresholds(self.sd.X_val, self.sd.Y_val)
        print("Done")
