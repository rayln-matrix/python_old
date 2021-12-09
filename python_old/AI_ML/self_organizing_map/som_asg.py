# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:13:57 2020

@author: WB
"""


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from minisom import MiniSom
from sklearn.cluster import KMeans

data_total_df = pd.read_csv('data_total.csv', index_col=[0])

aseg_data = data_total_df.iloc[:,5:]
label_data = data_total_df.iloc[:,1]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(aseg_data)
y = label_data.values

###  what's the best choice of x, y?
som = MiniSom(x=10, y=10, input_len=19,sigma=1.0, learning_rate=0.5)
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