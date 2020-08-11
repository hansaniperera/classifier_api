# import classifier as classifier
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# number of possible label values
labelled_classes = 2

# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(64, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))

model.add(Dense(labelled_classes, activation='softmax'))

# COMPILE
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# number of epochs to train the CNN
epochs = 10

# number of images to feed into the CNN for every batch
batch_size = 10

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'train',  # this is the target directory
    target_size=(300, 300),  # all images will be resized to 300x300
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True)

# this is a similar generator, for validation data
test_set = test_datagen.flow_from_directory(
    'validation',
    target_size=(300, 300),
    batch_size=1,
    class_mode='binary',
    shuffle=False)

# TRAINING
checkpoint = ModelCheckpoint("model_weights.h5", verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(
          training_set,
          steps_per_epoch=training_set.n // training_set.batch_size,
          epochs=epochs,
          validation_data=test_set,
          validation_steps=test_set.n // test_set.batch_size,
          callbacks=callbacks_list)

# serialize model structure to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)



