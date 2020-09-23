import keras
import numpy as np
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras import optimizers
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

from skimage import io

image_size = 32

def own_rel():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(2, 2),
                     activation='relu',
                     input_shape=(image_size, image_size, 3)))
    model.add(Dropout(0.1))

    model.add(Conv2D(16, kernel_size=(2, 2),
                     activation='relu'))
    model.add(Dropout(0.1))
    
    #model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    return model

#Preprocess image 
def transform_img_fn(images_raw):
    trans_images = []
    for i in images_raw:
        #io.imshow(i)
        #io.show()
        #convert to np array
        #image = img_to_array(i)

        #reshape to match network default size
        #image = i.reshape(vgg16.image_size, vgg16.image_size, 3)

        #image = preprocess_input(i)

        trans_images.append(i)
    return trans_images
