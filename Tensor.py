import os 
import matplotlib.pyplot

os.environ["TF_CPP_MIN_LOG_LEVEL"]=2
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Input, Dense
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices ("GPU")
tf.config.experimental.set_memory_growth (physical_devices[0], True),



