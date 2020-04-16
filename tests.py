
## Tensorflow keras
os.system("pip install -q tf-nightly")
os.system("pip install -U keras-tuner") #De https://github.com/keras-team/keras-tuner
from graph_layer import *
from graph_controleur import *
from graphviz import *
from input_model import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import models
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.losses
import os
##Python / Colab
from google.colab import files
from google.colab import drive
import os
# clear_output()
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.python import debug as tf_debug
from tensorflow.python.client import device_lib
## Math libraries
import numpy as np
import scipy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
##Images
from PIL import Image
import cv2
## Graph
from graphviz import render
from graphviz import Digraph,Graph
drive.mount('/content/drive')
os.system("cd '/content/drive/My Drive/TIPE'")
#Dataset
import random
import pathlib
import shutil
import time
#Debugage
from IPython.display import clear_output
#Hyperparameters tuning
from kerastuner import BayesianOptimization, Objective
os.system("clear")
print("START***********************************************************************")
os.system("free -h")

import tensorflow as tf
print("Tensorflow version " + tf.__version__)

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


tuner = BayesianOptimization(
    create_model,
    objective=Objective("custom_accuracy", direction="max"),
    max_trials=75,
    executions_per_trial=3,
    directory='Bayesian_optimization',
    project_name='Bayesian_libre_lim_nb_param_tau6'
)

dataset_train = ArtificialDataset(nom="Train").map(traitement,num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().prefetch(tf.data.experimental.AUTOTUNE)
dataset_val = ArtificialDataset(nom="Validation").map(traitement,num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().prefetch(tf.data.experimental.AUTOTUNE)

tuner.search(dataset_train,
             epochs=5,
             validation_data=dataset_val,
             callbacks=[tf.keras.callbacks.EarlyStopping(min_delta=1e-3)])

tuner.results_summary(num_trials=2)