# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 01:00:48 2025

@author: Umons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy
import os



import numpy as np
import h5py
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from scipy import signal
from sklearn.model_selection import train_test_split



class LoadDataset():
    def __init__(self, ):
        self.dataset_name = 'data'
        self.labelset_name = 'label'

    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col / 2)], dtype=complex)

        data_complex = data[:, :round(num_col / 2)] + 1j * data[:, round(num_col / 2):]
        return data_complex

    def load_iq_samples(self, file_path, dev_range, pkt_range):
        '''
        Load IQ samples from a dataset.
        INPUT:
            FILE_PATH is the dataset path.
            DEV_RANGE specifies the loaded device range.
            PKT_RANGE specifies the loaded packets range.
        RETURN:
            DATA is the laoded complex IQ samples.
            LABLE is the true label of each received packet.
        '''

        f = h5py.File(file_path, 'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1

        label_start = int(label[0]) + 1
        label_end = int(label[-1]) + 1
        num_dev = label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt / num_dev)

        print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
              str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')

        sample_index_list = []

        for dev_idx in dev_range:
            sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)

        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)

        label = label[sample_index_list]

        f.close()
        return data, label
    
    
def _normalization(data):
    
    s_norm = np.zeros(data.shape, dtype=complex)
    for i in range(data.shape[0]):
      sig_amplitude = np.abs(data[i])
      rms = np.sqrt(np.mean(sig_amplitude ** 2))
      s_norm[i] = data[i] / rms
    return s_norm

  
    
    

file_path_train = r'D:\IQ Data\LoRa_RFFI\\dataset\Train\dataset_training_aug.h5'
file_path_test = r'D:\IQ Data\LoRa_RFFI\dataset\Test\dataset_seen_devices.h5'

dev_range = range(0, 30)
pkt_range = range(0, 1000)

# Load training data
LoadDatasetObj = LoadDataset()
data_train, label_train = LoadDatasetObj.load_iq_samples(file_path=file_path_train, dev_range=dev_range, pkt_range=pkt_range)

# Shuffle the training data and labels
index = np.arange(len(label_train))
np.random.shuffle(index)
data_train = data_train[index, :]
label_train = label_train[index]

# One-hot encoding for labels
label_train = label_train - dev_range[0]
label_one_hot = to_categorical(label_train)

# Load testing data
pkt_range = range(0, 400)
data_test, label_test = LoadDatasetObj.load_iq_samples(file_path=file_path_test, dev_range=dev_range, pkt_range=pkt_range)

test_one_hot = to_categorical(label_test, len(np.unique(label_test)))


norm_data = _normalization(data_train)


i = norm_data.real
q = norm_data.imag

train_ = np.stack((i, q), axis=-1)


from keras.layers import Input, Lambda, ReLU, Add
from keras.models import Model,Sequential
from keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten, LeakyReLU, AveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam,RMSprop
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical



#%%1D CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(16, 3, activation='relu', input_shape=(8192, 2)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),  # Use GlobalMaxPooling1D to reduce time steps to 1
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(30, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='softmax')  # Assuming 3 classes in the output
])




# Learning rate scheduler
early_stop = EarlyStopping('val_loss', min_delta=0, patience=30)
reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=10, verbose=1)
callbacks = [early_stop, reduce_lr]

opt = RMSprop(learning_rate=0.001)
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt)


model.summary()

# Start training
history = model.fit(train_,
                  label_one_hot,
                  epochs=400,
                  shuffle=True,
                  validation_split=0.20,
                  verbose=1,
                  batch_size=32,
                  callbacks=callbacks)


#evaluating the model on test data


norm_test = _normalization(data_test)

i = norm_test.real
q = norm_test.imag

test = np.stack((i, q), axis=-1)


pred_prob = model.predict(test)

pred_label = pred_prob.argmax(axis=-1)

acc= accuracy_score(label_test, pred_label)
print("Accuracy: {:.2f}%".format(acc * 100))

classes = np.unique(label_test)

conf_mat = confusion_matrix(label_test, pred_label)

plt.figure()
sns.heatmap(conf_mat, annot=True,
        fmt='d', cmap='Blues',
        annot_kws={'size': 7},
        cbar=False,
        xticklabels=classes,
        yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted label', fontsize=12)
plt.ylabel('True label', fontsize=12)
plt.show()