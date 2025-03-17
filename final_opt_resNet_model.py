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
    
    
    
class ChannelIndSpectrogram():
    def __init__(self, ):
        pass

    def _normalization(self, data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude ** 2))
            s_norm[i] = data[i] / rms

        return s_norm

    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[0]
        x_cropped = x[round(num_row * 0.3):round(num_row * 0.7)]

        return x_cropped

    def _gen_single_channel_ind_spectrogram(self, sig, win_len=256, overlap=128):
        '''
        _gen_single_channel_ind_spectrogram converts the IQ samples to a channel
        independent spectrogram according to set window and overlap length.
        INPUT:
            SIG is the complex IQ samples.
            WIN_LEN is the window length used in STFT.
            OVERLAP is the overlap length used in STFT.
        RETURN:
            CHAN_IND_SPEC_AMP is the genereated channel independent spectrogram.
        '''
        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(sig,
                                 window='boxcar',
                                 nperseg=win_len,
                                 noverlap=overlap,
                                 nfft=win_len,
                                 return_onesided=False,
                                 padded=False,
                                 boundary=None)

        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)

        # Generate channel independent spectrogram.
        chan_ind_spec = spec[:, 1:] / spec[:, :-1]

        # Take the logarithm of the magnitude.
        chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec) ** 2)

        return chan_ind_spec_amp

    def channel_ind_spectrogram(self, data):
        '''
        channel_ind_spectrogram converts IQ samples to channel independent
        spectrograms.
        INPUT:
            DATA is the IQ samples.
        RETURN:
            DATA_CHANNEL_IND_SPEC is channel independent spectrograms.
        '''

        # Normalize the IQ samples.
        data = self._normalization(data)

        # Calculate the size of channel independent spectrograms.
        num_sample = data.shape[0]
        num_row = int(256 * 0.4)
        num_column = int(np.floor((data.shape[1] - 256) / 128 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i, :, :, 0] = chan_ind_spec_amp

        return data_channel_ind_spec
    
    
    

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



ChannelIndSpectrogramObj = ChannelIndSpectrogram()

data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_train)
print(data.shape)


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



def resblock(x, kernelsize, filters, first_layer=False):
    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)

        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        fx = BatchNormalization()(fx)

        x = Conv2D(filters, 1, padding='same')(x)

        out = Add()([x, fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)

        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        fx = BatchNormalization()(fx)
        #
        out = Add()([x, fx])
        out = ReLU()(out)

    return out


#6 million parameters
def classification_net(datashape, num_classes):
    datashape = datashape

    inputs = Input(shape=(np.append(datashape[1:-1], 1)))

    x = Conv2D(16, 4, strides=2, activation='relu', padding='same')(inputs)

    x = resblock(x, 3,16)

    x = resblock(x, 3, 8, first_layer=True)
    x = resblock(x, 3, 8)

    x = AveragePooling2D(pool_size=2)(x)

    x = Flatten()(x)

    x = Dense(512)(x)

    x = Lambda(lambda x: K.l2_normalize(x, axis=1), name='feature_layer')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model



# Learning rate scheduler
early_stop = EarlyStopping('val_loss', min_delta=0, patience=30)
reduce_lr = ReduceLROnPlateau('val_loss', min_delta=0, factor=0.2, patience=10, verbose=1)
callbacks = [early_stop, reduce_lr]


opt = RMSprop(learning_rate=1e-3)
model = classification_net(data.shape, len(np.unique(label_train)))
model.compile(loss='categorical_crossentropy', optimizer=opt)


model.summary()


# Start training
history = model.fit(data,
                  label_one_hot,
                  epochs=400,
                  shuffle=True,
                  validation_split=0.20,
                  verbose=1,
                  batch_size=32,
                  callbacks=callbacks)


#evaluating the model on test data

#create test data spectrogram
test_specs = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_test)
print(test_specs.shape)


pred_prob = model.predict(test_specs)

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