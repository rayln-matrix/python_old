# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:46:20 2020

@author: WB
"""


###### Self- Organized Mapping
## the relation between SOM / K-means clustering
## SOM: un-supervised learning

### K-mean clusetr algorithm:
# 1. set cluster numbers (Arm method)
# 2. Select random points to be centroids
# 3. Assigned each data points to the closes centroids (using Geometric distances)
# 4. Compute and place the new centroid of each cluster (Need a good algorithm)
# 5. Reassign each data point to the new closed centroid. 
#    --->If any reassignment took place, go to 4.
#    ---> else: go to finish 
## problem: initialization trap --> use K-means ++


## IN SOM: weights are characteristics of the nodes
#  ----> BMU: Best Matching Units (Nodes that are closet to the input data)
#  ----> How to update the weights around BMU? How to select the radius?
#  ----> radius reduced every epcho


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


data_set = pd.read_csv('Credit_Card_Applications.csv')
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:,-1].values

# Feature Scaling:
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# Import pre-existed SOM: minisom
from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len=15,sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train(data=X, num_iteration=100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
for i, x in enumerate(X):
    w = som.winner(x)
    # plotting the winning nodes: place the marker at the center of the winning node 
    # w[0]:x coordinates, w[1]:y coordinates
    plot(w[0]+0.5,w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor= 'None',
         markersize=10, markeredgewidth=2)
show()

# Return dict: keys--> coordinates
# Outliers: (9,6), (7,7)
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(9,6)],mappings[(7,7)]), axis=0)
frauds = sc.inverse_transform(frauds)

customers = data_set.iloc[:, 1:].values 
is_fraud = np.zeros(len(data_set))
for i in range(len(data_set)):
    if data_set.iloc[i,0] in frauds:
        is_fraud[i] = 1
        
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers) ## sc object is fit and can be apply to different data

from keras import Sequential
from keras.layers import Dense
### Create
classifier = Sequential()

#Input layer:
#classifier.add(keras.Input(shape=(14,)))

# First hidden layer: recieving input from input layer
classifier.add(Dense(2, activation='relu',kernel_initializer='glorot_uniform', input_dim=15))
# Output layer:
classifier.add(Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform'))
classifier.summary()
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

##  Fiting the network:
### at epoch 75, the accuracy reaches the maximum
### After scaling the training data, the performance reaches new high: 0.86--> try adding ephos
#----> new hight: 0.865
#---> What about changing the batch_size--> test for 10 (original) 20 ---> doesn't improve it 
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

## Predicting: can use the F9 to run single line
y_pred = classifier.predict(customers)
#y_pred_tf = (y_pred > 0.5)
y_pred = pd.concatenate((data_set.iloc[:,0:1],y_pred),axis=1)
y_pred = y_pred[y_pred[:,1].argsort()] #---> sorting the probabilties of fraud
### df.iloc[:, 0:1] --> can convert to 2d array