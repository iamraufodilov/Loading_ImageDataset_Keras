import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import tensorflow as tf

IMG_WIDTH=200
IMG_HEIGHT=200
batch_size=4

train_dir = 'G:/rauf/STEPBYSTEP/Data/IntelImageClassification/seg_train/seg_train'
test_dir = 'G:/rauf/STEPBYSTEP/Data/IntelImageClassification/seg_pred/seg_pred'
val_dir = 'G:/rauf/STEPBYSTEP/Data/IntelImageClassification/seg_test\seg_test'

import matplotlib.image as mpimg
from matplotlib import pyplot as plt

test_image='G:/rauf/STEPBYSTEP/Data/IntelImageClassification/seg_train/seg_train/buildings/0.jpg'
img = mpimg.imread(test_image)
plt.imshow(img)

image_gen_train = ImageDataGenerator(rescale=1./255,
                                     zoom_range=0.2,
                                     rotation_range=65,
                                     shear_range=0.09,
                                     horizontal_flip=True,
                                     vertical_flip=True)

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                     class_mode='sparse')

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='sparse')

my_class_names = train_data_gen.class_indices.keys()
print(my_class_names)

model=tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(200, 200, 3)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6)])

my_summary = model.summary()
print(my_summary)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data_gen,steps_per_epoch=len(train_data_gen)//batch_size, validation_data=val_data_gen, epochs=2)
