import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

train_path = 'casting_data/train/'
test_path = 'casting_data/test/'

image_size = (300,300,1)
batch_size = 32

image_gen = ImageDataGenerator(rescale=1/255)

train_set = image_gen.flow_from_directory(train_path,
                                          target_size=image_size[:2],
                                          color_mode="grayscale",
                                          batch_size=batch_size,
                                          class_mode='binary',
                                          shuffle=True)

test_set = image_gen.flow_from_directory(test_path,
                                         target_size=image_size[:2],
                                         color_mode="grayscale",
                                         batch_size=batch_size,
                                         class_mode='binary',
                                         shuffle=False)


model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(3,3),input_shape=image_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=8, kernel_size=(3,3),input_shape=image_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(224))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint('model/model1.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              patience=2, 
                              min_lr=0.00001, 
                              mode='auto')


callbacks = [checkpoint, reduce_lr]
results = model.fit_generator(train_set,
                              epochs=10,
                              validation_data=test_set,
                              callbacks=callbacks)
