#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import tensorflow.keras as keras
data_dir = "prepped_data/train/"


# In[7]:


import tensorflow as tf
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True




from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda, concatenate
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', 
                 input_shape=(128, 128, 2)
                 
                ))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))              
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(.5))

def slice_tensor(i):
    return Lambda(lambda x: x[:, :, :, i:i + 1])
x = Input((128, 128, 4))

a = concatenate([slice_tensor(0)(x), slice_tensor(2)(x)], -1)
b = concatenate([slice_tensor(1)(x), slice_tensor(3)(x)], -1)
a = model(a)
b = model(b)

y = concatenate([a, b])


y = Dense(1024, activation='relu')(y)
y = Dropout(.5)(y)
y = Dense(12)(y)

model = keras.Model(x, y)
model.summary()

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=Adam(lr=.0002))



# In[9]:

"""import ultrasoundgeneration
import importlib
importlib.reload(ultrasoundgeneration)


tdata, tclasses = ultrasoundgeneration.load_dataset(ultrasoundgeneration.test_volumes_path)


# In[13]:


while True:
    data, classes = ultrasoundgeneration.load_dataset(ultrasoundgeneration.train_volumes_path)
    model.fit(data, 
          classes,
          batch_size=90,
          epochs=1,
          verbose=1,
          validation_data=(tdata[:256], tclasses[:256]))


# In[ ]:


model.load_weights("fake_ultrasound_model")


# In[24]:




model.compile(loss=keras.losses.mean_squared_error,
              optimizer=Adam(lr=.0002))


# In[11]:


rclasses = model.predict(tdata)


# In[14]:


tdata.shape


# In[10]:


tclasses[0]


# In[12]:


import matplotlib.pyplot as plt
for j in range(12):
    plt.scatter(rclasses[:, j], tclasses[:2000, j], s=.1)
    plt.show()


# In[20]:


model.save("fake_ultrasound_model3")


# In[3]:


import matplotlib


# In[4]:


get_ipython().run_line_magic('pinfo', 'plt.scatter')


# In[4]:


tclasses[:2000, j]import tensorflow
tensorflow.test.is_gpu_available()


# In[29]:


import tensorflow as tf
with tf.Session() as sess:
    devices = sess.list_devices()


# In[1]:


import sys
sys.executable


# In[38]:


#with open("prepped_data/chunk0.pickle", 'wb') as c0:
#    pickle.dump([data, classes], c0, protocol=4)


# In[3]:


import pickle
data, classes = pickle.load(open("prepped_data/chunk0.pickle", "rb"))


# In[42]:


data.dtype


# In[41]:


import matplotlib.pyplot as plt
for i in range(4):b 
    plt.imshow(tdata[np.random.randint(tdata.shape[0]), :, :, i] / 10 + 1)
    plt.colorbar()
    plt.show()


# In[16]:


tdata.shape


# In[25]:


error = tclasses - rclasses


# In[34]:


np.sum(np.abs(error[37]))


# In[1]:


import julia.api
jl = julia.api.Julia(bindir="C:/Users/hastings.local/AppData/Local/Julia-1.3.0-alpha/bin/julia.exe")


# In[ ]:





# In[ ]:




"""
