#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import tensorflow.keras as keras
data_dir = "prepped_data/train/"


# In[7]:


import tensorflow as tf
#config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
#config.gpu_options.allow_growth = True

import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', 
                 input_shape=(128, 128, 2)
                 
                ))
model.add(BatchNormalization(momentum=.9))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization(momentum=.9))


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization(momentum=.9))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization(momentum=.9))
#model.add(MaxPooling2D(pool_size=(2, 2)))              
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization(momentum=.9))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization(momentum=.9))
#model.add(Dropout(.5))

def slice_tensor(i):
    return Lambda(lambda x: x[:, :, :, i:i + 1])
x = Input((128, 128, 4))

a = concatenate([slice_tensor(0)(x), slice_tensor(2)(x)], -1)
b = concatenate([slice_tensor(1)(x), slice_tensor(3)(x)], -1)
a = model(a)
b = model(b)

c = Input((3,))

y = concatenate([a, b, c])


y = Dense(1024, activation='relu')(y)
#y = Dropout(.5)(y)
y = Dense(3)(y)

model = keras.Model((x, c), y)
model.summary()

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=Adam(lr=0.0002))

