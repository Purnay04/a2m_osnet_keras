import os
import types
from datetime import datetime
from pathlib import Path


config = types.SimpleNamespace()

HOME_DIR = str(Path(os.getcwd()))
NOW = str(datetime.date(datetime.now()))



# Subdirectory name for saving trained weights and models
config.SAVE_DIR = os.path.join(HOME_DIR, 'checkpoints', NOW)

# Subdirectory name for saving TensorBoard log files
config.LOG_DIR = os.path.join(HOME_DIR, 'logs', NOW)

# Default path to the ImageNet TFRecords dataset files
config.DATASET_DIR = os.path.join('/media/toanmh/Workspace/Datasets/ImageNet/ILSVRC2012', 'tfrecords')

# Number of parallel works for generating training/validation data
config.NUM_DATA_WORKERS = 8

# Do image data augmentation or not
config.DATA_AUGMENTATION = True

# 
config.TRAIN_DIR = os.path.join('/media/toanmh/Workspace/Datasets/ImageNet/ILSVRC2012', 'train')
config.VALIDATION_DIR = os.path.join('/media/toanmh/Workspace/Datasets/ImageNet/ILSVRC2012', 'validation')
config.OUT_DIR = os.path.join('/media/toanmh/Workspace/Datasets/ImageNet/ILSVRC2012', 'tfrecords')