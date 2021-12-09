# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:06:22 2020

@author: WB
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
#from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from keras.layers import Dropout
 
### Read the feature:
raw_data = pd.read_csv('Churn_Modelling.csv')
X = raw_data.iloc[:,3:-1]
y = raw_data.iloc[:,-1].values
#print(X['Geography'].unique()) ---> Three countries

### Use LabelEncoder to encode categorical variables --> dummy variables
label_encoder_1=LabelEncoder()
label_encoder_2=LabelEncoder()
X.iloc[:,1] = label_encoder_1.fit_transform(X.iloc[:,1])
X.iloc[:,2] = label_encoder_2.fit_transform(X.iloc[:,2])
#print(X)---> map country to [0,1,2] and gender to [0,1]

### When use ColumnTransformer / OneHotEncoder,the categorical feature will be transform 
#   into binary dummy features: ex: {country: [JP,CA,USA]}--> {1:1/0, 2:1/0, 3:1/0}
### So, If I use drop, there will be: 
#       3 + 2 =5 feature: from ['France','Spain','Germany'] and ['male','female']
# If passthrough: 11-2+5=14
### Female = 1, male = 0, 
### 0: France, 1: Spain, 2: German, 3: gender
ctf = ColumnTransformer([('Geography',OneHotEncoder(),['Geography']),
                        ('Gender',OneHotEncoder(),['Gender'])],remainder='passthrough')
X_trans = ctf.fit_transform(X)
### This does not avoid dummy variable trap!!!


### Dummy features first and then the original data feature
#print(X.iloc[0,:])  ---> checking the original feature
#print(X_trans[0])   ---> Matched
#print(y)            ---> label is OK

 
X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train) ## sc object is fit and can be apply to different data
X_test = sc.transform(X_test) ## use the sc (from X_train) to transform X_test (why don't use different?)
print(X_train)
print(X_test)
print(y_train)
print(y_test)


### Create
classifier = Sequential()

#Input layer:
#classifier.add(keras.Input(shape=(14,)))

# First hidden layer: recieving input from input layer
classifier.add(Dense(6, activation='relu',kernel_initializer='glorot_uniform', input_dim=13))#, name='Layer1'))

# Second hidden layer: recieving input from first hidden layer
classifier.add(Dense(6, activation='relu',kernel_initializer='glorot_uniform')) #, name='Layer2'))

# Third hidden layer: recievinf input from second layer and output to output layer
classifier.add(Dense(6,activation='relu', kernel_initializer='glorot_uniform')) #, name='Layer3'))
## Adding a layer doesnot help to improve accuracy

# Output layer:
classifier.add(Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform'))#, name='output'))

# Testing Network propperties:
#print(len(classifier))
#classifier.pop()
#print(len(classifier))

## Showing the summary of the ANN:
classifier.summary()

## Compile:optimizer='adam',loss='binary_crossentropy', metrics='accuracy'
# Why need metric? if there is loss?
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

##  Fiting the network:
### at epoch 75, the accuracy reaches the maximum
### After scaling the training data, the performance reaches new high: 0.86--> try adding ephos
#----> new hight: 0.865
#---> What about changing the batch_size--> test for 10 (original) 20 ---> doesn't improve it 
classifier.fit(X_train, y_train, batch_size=10, epochs=250)

## Predicting: can use the F9 to run single line
y_pred = classifier.predict(X_test)
y_pred_tf = (y_pred > 0.5)

#confustion_m = confusion_matrix(y_test, y_pred_tf)

single_test_data_list=[1,0,0,0,1,600,40,3,60000,2,1,1,50000]
single_test_data_array=np.array(single_test_data_list).reshape(-1,13)

predict_single=classifier.predict(sc.transform(single_test_data_array))
predict_single_tf=(predict_single > 0.5)
####### Question:
# What is  non-linear topology (e.g. a residual connection, a multi-branch model)?

#### Using Cross_validation model: to solve the bias-variance balance problem
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, activation='relu',kernel_initializer='glorot_uniform', input_dim=13))
    #classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(6, activation='relu',kernel_initializer='glorot_uniform'))
    classifier.add(Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

#classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=250)
#### Arg: n_job: number of cpu to use: set -1 --> use all cpus in parallel
#---> how this is implement? (by c language?) 
#accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
#accury_mean = accuracies.mean()
#accury_variance = accuracies.std()



#### Drop_out regularization / Parameters tuning
# Overfitting: accuracy is high when test data is highly correlated with the training set--> variance of cross-valid is high
# Drop out: random dissable of nerons
####---> drop out rate: 0.1~ 0.5  above 0.5 is possible to underfitting
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25,32], # according to the experience of the tutor
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']
              }
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters, scoring='accuracy', cv=10)
grid_search =  grid_search.fit(X_train,y_train)
best_parameter = grid_search.best_params_
best_accuracy = grid_search.best_scores_