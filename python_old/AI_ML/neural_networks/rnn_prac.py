# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:49:53 2020

@author: WB
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#### Data preprocessing:

data_train_df = pd.read_csv('Google_Stock_Price_Train.csv')
data_train_set = data_train_df.iloc[:,1:2].values

#print(type(data_train_set))

## Standardization / normalization ---> recommend: sigmoid function + normalization

sc = MinMaxScaler(feature_range=(0,1))
scaled_train_data = sc.fit_transform(data_train_set)

## Defining timestep---> importnat for the LSTM to remember
# -->EX: 60, using 60  pasted days' prices : t0-t59 to predict t60

X_train = []
y_train = []

for i in range(60,1257):
    X_train.append(scaled_train_data[i-60:i,0])
    y_train.append(scaled_train_data[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

## Using reshape to add new dimension (indicator) and match the RNN input structure
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
# 1: number of indicators


#### Building RNN:
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
# First layer:
regressor.add(LSTM(units=50,return_sequences=True, input_shape=(X_train.shape[1],1))) 
regressor.add(Dropout(0.2))
# how to choose the numbers of units??

#  Dropout rate: 0.2 / return_sequences: recurrent 
# Second layer:
regressor.add(LSTM(units=50,return_sequences=True)) 
regressor.add(Dropout(0.2))

# Third layer:
regressor.add(LSTM(units=50,return_sequences=True)) 
regressor.add(Dropout(0.2))

# Additional: for testing
regressor.add(LSTM(units=50,return_sequences=True)) 
regressor.add(Dropout(0.2))

## This is for further testing:
#regressor.add(LSTM(units=50,return_sequences=True))
#regressor.add(Dropout(0.2))

# Second layer:
regressor.add(LSTM(units=50,return_sequences=False)) 
regressor.add(Dropout(0.2))

# Output: Full connected
regressor.add(Dense(units=1))

regressor.summary()

regressor.compile(optimizer='adam', loss='mean_squared_error')


# Fitting data:
regressor.fit(X_train, y_train, epochs=100, batch_size=32)
# (100,32) --> (200, 64)


### Test data:
data_test_df = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = data_test_df.iloc[:,1:2].values

# Beware of the scaling:
data_total_df = pd.concat((data_train_df['Open'], data_test_df['Open']),axis=0 )
#test_df= data_train_df['Open'].append(data_test_df['Open']) --> Same as above
inputs = data_total_df[len(data_total_df)-len(data_test_df)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test= np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

### Predicted priced and re-scaled
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


### Visualizing:
plt.plot(real_stock_price, color = 'red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

#### testing result:
# (100,32)-->(200,64): Loss:0.0013
# adding one more LSTM layer: Losss: 0.0017
# (200,64) & Adding one more LSTM layer Loss:0.0012
## Futher testing:
# dropout / more LSTM
