# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:45:19 2020

@author: WB
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import random
from sklearn.model_selection import train_test_split


####  Checking GPU:
#from tensorflow.python.client import device_lib
#from keras import backend
#print(len(backend.tensorflow_backend._get_available_gpus()))
#print(device_lib.list_local_devices())


#### Checking the package: read files / get np.array data
data_path = 'D:\python_accon\Data_3DCNN\gzipped\subsample'#'\subject0001'


#example1=os.path.join(data_path, 'subsam3_3DICAmap_all.nii')
#### Testing data
#img = nib.load(example1)
#print(img.get_data_dtype())
#data = img.get_fdata()
#print(data)
#i=1
#for colm in data:
#    for col2 in colm:
#        print(i)
#        print(col2)
#        i += 1 


#nii_data_dict=dict()
nii_data_list=[]
for roots, dirs,files in os.walk(data_path):
   for f in files:
        sub_name=roots[-11:]
        nii=os.path.join(roots, f)
        img = nib.load(nii)
        data = img.get_fdata()
        #nii_data_dict[sub_name]=data
        nii_data_list.append(data)
        print(sub_name)
#print(nii_data_dict)        
#### Batch or single file?
##  Batch: faster after the first one
label_data = pd.read_csv('D:\python_accon\Data_3DCNN\data_list_new.txt',sep=' ', index_col=None, header= None)
label_data = label_data[[0,2]].rename(columns={0:'Subject',2:'label'})
y_data = label_data['label'].values
#print(label_data[label_data['label']==1]['label'].count()) ### 188
#print(label_data[label_data['label']==0]['label'].count()) ### 195
#subject_list = [key for key in nii_data_dict.keys()]

## Using the random to sample subject for train and test:
#train_index=random.sample(range(0,383),345)
#test_index=[idx for idx in range(0,383) if not(idx in train_index)]
#train_subject = [subject_list[i] for i in train_index]
#test_subject = [subject_list[i] for i in test_index]
## Train / Test data:
#data_df = pd.DataFrame({'Subject':nii_data_dict.keys(),'Data':nii_data_dict.values()})
#total_df = pd.merge(data_df, label_data, on='Subject')
#X_train = total_df[total_df['Subject'].isin(train_subject)]['Data'].to_numpy()
#y_train = total_df[total_df['Subject'].isin(train_subject)]['label'].values
#X_test = total_df[total_df['Subject'].isin(test_subject)]['Data']
#y_test = total_df[total_df['Subject'].isin(test_subject)]['label']
#data_total = np.array(nii_data_list)
        
X_train, X_test, y_train, y_test = train_test_split(nii_data_list, y_data, test_size=0.2, random_state=0)
# Train: 306
# Test: 

from keras.models import Sequential
from keras.layers import Convolution3D
from keras.layers import MaxPooling3D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization        
from keras.layers import Dropout

### input shape = (batchsize(=t?), volume(64*64*43), channel(1?))
data_shape = (16, 64, 64, 43)

model = Sequential()

## First set:
model.add(Convolution3D(64,kernel_size=(3,3,3), strides=(1,1,1),activation='relu', kernel_initializer='glorot_uniform', input_shape=data_shape))
model.add(Convolution3D(64,kernel_size=(3,3,3), strides=(1,1,1),activation='relu', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True,scale=True))
model.add(MaxPooling3D(pool_size=(2,2,2)))

## Second set:
model.add(Convolution3D(128,kernel_size=(3,3,3), strides=(1,1,1),activation='relu', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True,scale=True))
model.add(Convolution3D(128,kernel_size=(3,3,3), strides=(1,1,1),activation='relu', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True,scale=True))
model.add(MaxPooling3D(pool_size=(2,2,2)))

## Flatten()
model.add(Flatten())

## Full connected layer:
model.add(Dense(4096, activation='relu'))
model.add(Dropout(rate=0.7))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(rate=0.7))
model.add(Dense(1, activation='sigmoid'))
        
## Compile:
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## Model Summary:
model.summary()


print(len(X_train))
print(len(X_test))
#X_train = np.array(X_train)
## Fitting data:
#model.fit(np.array(X_train), y_train, batch_size=15,epochs=50)


X_test = np.array(X_test)
from keras.models import load_model
model2 = load_model(r'D:\python_accon\Data_3DCNN\3dcnn_model_1')
#########################################################################

## Parameters and hardware in paper:
# GPU: GTX 1080 Ti, 
# Dropout: 0.7, in fully connected layers
# Batch size: 12, epchos: 50
# learning rate: f [1, 0.1, 0.01, 0.001, 0.0001, and 0.00001] 
# epsilon: [0.1 : 0.05 : 1]
# dropout rate: 0.7 
# Optimizar: adam
        