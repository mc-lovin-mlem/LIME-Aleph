from tensorflow import keras

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import own_rel
import numpy as np

image_size = own_rel.IMAGE_SIZE
train_dir = '../../datasets/single_relation/train'
val_dir = '../../datasets/single_relation/val'

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", default=10, help="The number of epochs to train.")
args = vars(ap.parse_args())

#Create the model
model = own_rel.own_rel()

#Compile model
model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ['acc'])

#Show model summary, check trainable parameters
model.summary() 

#Setup data generators
train_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)

epochs = int(args['epochs'])

#Batch size
train_batchsize = 64
val_batchsize = 64

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (image_size, image_size),
    batch_size = train_batchsize,
    class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size = (image_size, image_size),
    batch_size = val_batchsize,
    class_mode = 'categorical',
    shuffle = False)

#Early Stopping
callbacks = [ModelCheckpoint(filepath = '../../models_to_explain/model.h5', 
                             monitor = 'val_loss',
                             save_best_only = True),
			 EarlyStopping(monitor = 'val_loss', patience=3, restore_best_weights=True)]

#Train model
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples/train_generator.batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples/validation_generator.batch_size,
    verbose = 1,
    callbacks = callbacks)
