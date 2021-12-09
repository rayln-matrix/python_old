# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 09:32:11 2020

@author: WB
"""


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#### Convolution: Feature detector, stride size, reLU

# Establishing the model object
classifier = Sequential()

# Convolutional layer: consists of different feature maps
### Color image: 3D array, Black-white: 2D array: input_shape(numb_of_channel, size-l, size-w)
### If working on CPU, number of feature detectors should not be too large (here use 32, size: 3*3)
### same reason: input_shape should not be too large (3d, size:64*64)-->this is the format of Theano backend
### ---> tensorflow backend:input_shape=size, channels
### Q: how Convolution2D establish feature detector?
classifier.add(Convolution2D(64,(3,3),input_shape=(64,64,3),activation='relu'))
### -->Result:　Using 64 > 32 feature detectors reduces the accuracy between training and testing
### --> How the keras.convolution2D select or create feature detectors?


### Pooling: Size of the pooled feature map: (d+1)/2 or d/2 , d is the size of original feature map
### Pool size 2*2 is usually useful
classifier.add(MaxPooling2D(pool_size=(2,2)))

## Adding more convolution layer:
classifier.add(Convolution2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

## Adding another convolution layer:
classifier.add(Convolution2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


### Flattening: transform pooled feature map to single vector
### Why not direct flattening the image: 
### it could not keep the information of the relation between a pixel and other pixels around
classifier.add(Flatten())

### Full connected layer:
### Q: what is the size of input (output of Flatten()?---> all pooled feature map?)
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

### Compile:
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


### Image preprocessing:
### Image augmentation: using rotate, shear,zoom,flip..etc...to enrich dataset
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,  # all pixel value will be between 0 and 1
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), ## expected size in CNN, here is 64*64
        batch_size=32,
        class_mode='binary')
### Success: Found 8000 images belonging to 2 classes.

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        samples_per_epoch=8000, ## The amount of traing data
        nb_epoch=25,
        validation_data=test_set,
        nb_val_samples=2000 )# the amount of test data)


### Deeper can help?  how to proof it works? ---> Adding Convolution layer or FullyConnected layer?  
### Can I print the parameters of convolutional layer? 

'''import numpy as np
from keras.preproccessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction ='dog'
else:
    prediction = 'cat'
'''

