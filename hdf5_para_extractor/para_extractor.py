from keras.datasets import mnist
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
import keras as K
import keras
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
import numpy as np
import glob
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
import h5py


model = load_model('963-0.99064.hdf5', compile=False)  # model 폴더에서 best epoch 뽑아서 파일 이름 적기


print('\n======== Weights of Hidden Layer1 ========')
print(model.get_weights()[0])
print('\n======== Weights of Hidden Layer2 ========')
print(model.get_weights()[2])
print('\n======== Weights of Hidden Layer3 ========')
print(model.get_weights()[4])

print('\n======== biases of Hidden Layer1 ========')
print(model.get_weights()[1])
print('\n======== biases of Hidden Layer2 ========')
print(model.get_weights()[3])
print('\n======== biases of Hidden Layer3 ========')
print(model.get_weights()[5])



np.savetxt("layer_weights1.txt",model.get_weights()[0], fmt = '%f', delimiter = ',')
np.savetxt("layer_bias1.txt"   ,model.get_weights()[1], fmt = '%f', delimiter = ',')
np.savetxt("layer_weights2.txt",model.get_weights()[2], fmt = '%f', delimiter = ',')
np.savetxt("layer_bias2.txt"   ,model.get_weights()[3], fmt = '%f', delimiter = ',')
np.savetxt("layer_weights3.txt",model.get_weights()[4], fmt = '%f', delimiter = ',')
np.savetxt("layer_bias3.txt"   ,model.get_weights()[5], fmt = '%f', delimiter = ',')




