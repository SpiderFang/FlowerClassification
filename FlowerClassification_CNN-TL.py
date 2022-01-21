
# coding: utf-8

# In[1]:


## Using Transfer Learning technique to do images classification ##
# """
# Workaround Notes:
# 1. Q: the validation_split argument in ImageDataGenerator not supported in Keras 2.1.3(server version)
#    A: upgrade to the latest Keras(version 2.2.2): pip install keras --upgrade
# 2. Q: Activation "softmax" in the latest Keras(version 2.2.2) not matched TensorFlow 1.4(server version)
#    A: change Activation "softmax" to tf.nn.softmax
# => Keras 2.1.5 is exactly for tensorflow 1.4.1! Instead of using "pip install keras==2.1.5" to overcome both Q1&Q2.
#
# Experimental Result:
# Keras 2.1.5 + tensorflow 1.4.1 got better accuracy than Keras 2.2.2 + tensorflow 1.4.1
# """


# In[2]:


import numpy as np
import pandas as pd
import cv2


# In[3]:


import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

TRAIN_IMG_DIR = "./train/" #training_set at ./train/
TEST_IMG_DIR = "./test/" #testing_set at ./test/testimg/

NUM_CLASSES = 5 #target labels(ground truth), total 5 classes(check mapping.txt)

# Image shapes
IMG_WIDTH = 224
IMG_HEIGHT = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, CHANNELS)

BATCH_SIZE = 16
EPOCHS = 100

## Loading pre-trained network models in Keras
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.mobilenet import MobileNet
# from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
# from keras.applications.densenet import DenseNet169
# from keras.applications.densenet import DenseNet201

## Setting pre-trained network models
DenseNet_model = DenseNet121(include_top = False, weights = "imagenet", input_shape = INPUT_SHAPE)
conv_base = DenseNet_model

conv_base.trainable = False

## Create our model based on the pre-trained network model
model = Sequential()
model.add(conv_base) #comes from the pre-trained model

#Fully-connected NN layers
#fully-connected 1st layer
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#fully-connected final layer
model.add(Dense(NUM_CLASSES, activation="softmax"))
# !change Activation from keras to tf.nn.softmax, because TF version too old on Server!
# model.add(Dense(NUM_CLASSES))
# import tensorflow as tf
# model.add(Activation(tf.nn.softmax))

# Freeze the base model before model.compile
# conv_base.trainable = False
# for layer in conv_base.layers:
#     layer.trainable = False

# opt_adam = optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
# opt_rmsprop = optimizers.RMSprop(lr=1e-5, decay=0.01)
model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

print (model.summary())


# In[4]:


## Using Keras ImageDataGenerator to load images batch and do data augmentation on the fly.
#!validation_split argument not supported in Keras 2.1.3(server version)!
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        validation_split = 0.20
)

valid_datagen = ImageDataGenerator(
        rescale = 1./255,
#         rotation_range = 20,
#         width_shift_range = 0.2,
#         height_shift_range = 0.2,
#         shear_range = 0.2,
#         zoom_range = 0.2,
#         horizontal_flip = True,
        validation_split = 0.20
)

test_datagen = ImageDataGenerator(rescale = 1./255)

## Using Keras datagen.flow_from_directory to load images from every sub-directories at train,(validation),test directory
train_generator = train_datagen.flow_from_directory(
        directory = TRAIN_IMG_DIR,
        target_size = (IMG_WIDTH, IMG_HEIGHT),
        color_mode = "rgb",
        batch_size = BATCH_SIZE,
        class_mode = "categorical",
        shuffle = True,
        seed = 33,
        subset = "training"
)

validation_generator = valid_datagen.flow_from_directory(
        directory = TRAIN_IMG_DIR,
        target_size = (IMG_WIDTH, IMG_HEIGHT),
        color_mode = "rgb",
        batch_size = BATCH_SIZE,
        class_mode = "categorical",
        shuffle = True,
        seed = 33,
        subset = "validation"
)

test_generator = test_datagen.flow_from_directory(
        directory = TEST_IMG_DIR,
        target_size = (IMG_WIDTH, IMG_HEIGHT),
        color_mode = "rgb",
        batch_size = 1,
        class_mode = None,
        shuffle = False
)

## Amounts of individual set: training, validation, test
print (train_generator.n) #amounts of train_generator
print (validation_generator.n) #amounts of validation_generator
print (test_generator.n) #amounts of test_generator

## Labels from Keras data generator
print (train_generator.class_indices)
print (validation_generator.class_indices)

## Image shape check
print (train_generator.image_shape)
print (validation_generator.image_shape)
print (test_generator.image_shape)


# In[ ]:


## Fitting/Training the model
STEPS_PER_EPOCH = train_generator.n // BATCH_SIZE
VALIDATION_STEPS = validation_generator.n // BATCH_SIZE

# Callbacks setting
FILE_PATH = "./checkpoint-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5"
# EarlyStop = EarlyStopping(monitor="val_acc", patience=50, verbose=1, mode="max")
Checkpoint = ModelCheckpoint(FILE_PATH, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
# Callback_list = [EarlyStop, Checkpoint]
Callback_list = [Checkpoint]

history = model.fit_generator(
                generator = train_generator,
                steps_per_epoch = STEPS_PER_EPOCH,
                epochs = EPOCHS,
                callbacks = Callback_list,
                validation_data = validation_generator,
                validation_steps = VALIDATION_STEPS,
                shuffle = True
)

## Evaluate the model
# model.evaluate_generator(generator = )

## Predict the test set, then we'll get a probability nparray
test_generator.reset()
pred_probability = model.predict_generator(test_generator, verbose=1)


# In[ ]:


## Convert the prediction probability nparray to pandas dataframe to understand its structure
df_pred = pd.DataFrame(pred_probability)
display(df_pred)


# In[ ]:


# """
# This section is for saving the results to the CSV file.
# """
## Get the predicted class indices from model prediction result.(we can check it from the above probability dataframe)
predicted_class_indices = np.argmax(pred_probability, axis=1)

#default labels from Keras data generator(ie. names of sub-directories of training set)
keras_labels = (train_generator.class_indices)
#get the names of class labels
keras_labels_swap = dict((value, key) for key, value in keras_labels.items())
class_name = [keras_labels_swap[idx] for idx in predicted_class_indices]

## Reading pre-defined labels from mapping.txt, and store it to a dictionary
mapping = {}
with open("./mapping.txt") as f:
    for line in f:
        (key, val) = line.split(sep=",")
        mapping[str(key)] = int(val)

## Because predicted_class_indices come from Keras (data generator) default labels,
## this may not match our pre-defined labels (from mapping.txt).
## I use pandas.Series.map(arg=Dict) to remap predicted_class_indices to pre-defined labels.
ps = pd.Series(data = class_name)
class_predictions = ps.map(mapping)

## Get filenames of all test images
files = test_generator.filenames #!this output will include the directory path name!
#use string.strip() to retrieve exact filename(without directory path name) of test images
filenames = []
for num in range(len(files)):
    lst = files[num].lstrip("testimg/").rstrip(".jpg")
    filenames.append(lst)

## Save the results to the csv file
results = pd.DataFrame({"id" : filenames,
                        "class_name" : class_name,
                        "class" : class_predictions})
results.to_csv("results.csv", index=False)

submission = pd.DataFrame({"id" : filenames,
                           "class" : class_predictions})
submission.to_csv("submission.csv", index=False)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script FlowerClassification_CNN-TL.ipynb')


# In[ ]:


# """
# Experiment Results:
# Transfer Learning:
# DenseNet121---
# 1. Freeze the base model before model constructed and model.compile using:
#    conv_base.trainable = False
# -> val_acc = 0.85805
# 2. Freeze the conv_base model and layers before model.compile using:
#     conv_base.trainable = False
#     for layer in conv_base.layers:
#         layer.trainable = False
# -> val_acc = 0.85677
# 3. Freeze layers before model constructed and model.compile using:
#     for layer in conv_base.layers:
#         layer.trainable = False
# -> val_acc = 0.86
# 4. Freeze layers after model constructed before model.compile using:
#     for layer in conv_base.layers:
#         layer.trainable = False
# -> val_acc = 0.94
# When turning off earlystop, at epoch:70~75, val_acc:0.94 at most.
#
# Conclusion: Freeze all layers in base model after model constructed before model.compile will get better accuracy.
# ?Problem: Why setting "base_model.trainable = False" not work well? seems not broadcast to whole model?
# """

