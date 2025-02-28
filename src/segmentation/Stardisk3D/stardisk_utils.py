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
from stardist.models import Config3D, StarDist3D, StarDistData3D
from stardist import relabel_image_stardist3D, Rays_GoldenSpiral, calculate_extents
from stardist.matching import matching_dataset
import subprocess
import shutil
from tqdm import tqdm 
import csv
from glob import glob
import time
import tensorflow as tf


def filter_files(file_list, filter_list):
    filtered_list = []
    for fname in file_list:
        if b_any(fname.split('==')[0] in s for s in filter_list):
            filtered_list.append(fname)
    return filtered_list

def check_gpu():
    if tf.test.gpu_device_name()=='':
        print('You do not have GPU access.')

    else:
        print('You have GPU access')
        # Run the `nvidia-smi` command using subprocess to check GPU details
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi command not found. Ensure NVIDIA drivers are installed and the GPU is accessible.")

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

