# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:59:13 2020

@author: WB
"""


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

#### Use aparc volume/thickness as features, target: find the clusters


#### Read all the feature data
all_feature_df = pd.read_csv('all_new.csv')
roi_data = all_feature_df.iloc[:,5:].values
label_data = all_feature_df.iloc[:,1].values


#### Scaling the data: range-->(0,1)
sc = MinMaxScaler(feature_range=(0,1))
scaled_roi_data = sc.fit_transform(roi_data)

### Using AgglomerativeClustering to find the best cluster number:
hierachy_cluster = AgglomerativeClustering()
cluster1 = hierachy_cluster.fit_predict(scaled_roi_data)
h_cluster_df = pd.DataFrame(cluster1)

markers=['8','s','p','x']
#fig = plt.figure()
#ax = fig.add_subplot(111)
'''for col in all_feature_df.columns.tolist()[2:]:
    plt.scatter(all_feature_df['dx'], all_feature_df[col],c=h_cluster_df[0],  s=50)
    #plt.set_title('Cluster 1')
    #plt.set_xlabel('dx')
    #plt.set_ylabel(col)
    plt.colorbar(scatter)
    plt.show()'''


#### New data: adding cortex thickness: aparc.a2009s.thickness
hc_sz_train_df = pd.read_csv('hc_sz_train.csv')
hc_sz_test_df = pd.read_csv('hc_sz_test.csv')
roi_name = hc_sz_train_df.columns.tolist()[5:]
hc_sz_train_data = hc_sz_train_df[roi_name].values
hc_sz_train_label = hc_sz_train_df['dx'].values
hc_sz_test_data = hc_sz_test_df[roi_name].values
hc_sz_test_label = hc_sz_test_df['dx'].values

print(hc_sz_train_df['age'].mean())
age_range = [18,20,30,40,50]
for age in age_range:
    if age != 50:
        hc_sz_in_age_range = hc_sz_train_df[(hc_sz_train_df['age'] >=age) & (hc_sz_train_df['age'] < (age+10))]
        #print(len(hc_sz_in_age_range))
        sc = MinMaxScaler(feature_range=(0,1))
        scaled = sc.fit_transform(hc_sz_in_age_range[roi_name].values)
        cluster = hierachy_cluster.fit_predict(scaled)
        cluster= pd.DataFrame(cluster)
        hc_sz_in_age_range['cluster'] = cluster
        #print(len(cluster))
        for i in range(len(hc_sz_in_age_range)):
            hc_sz_in_age_range['cluster'].iloc[i]=cluster[0].iloc[i]
        #print(hc_sz_in_age_range)
        hc= hc_sz_in_age_range[hc_sz_in_age_range['dx']==0]['cluster']
        sz= hc_sz_in_age_range[hc_sz_in_age_range['dx']==1]['cluster']
        print(len(hc))
        print(len(sz))
        
        '''fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter=ax.scatter(hc_sz_in_age_range['dx'], hc_sz_in_age_range['age'],c=cluster[0],  s=50)
        ax.set_title('Cluster 1')
        ax.set_xlabel('dx')
        ax.set_ylabel('age')
        plt.colorbar(scatter)'''
    if age==50:
        hc_sz_in_age_range = hc_sz_train_df[hc_sz_train_df['age'] >=age]
        print(len(hc_sz_in_age_range))
        sc = MinMaxScaler(feature_range=(0,1))
        scaled = sc.fit_transform(hc_sz_in_age_range[roi_name].values)
        cluster = hierachy_cluster.fit_predict(scaled)
        cluster= pd.DataFrame(cluster)
        '''fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter=ax.scatter(hc_sz_in_age_range['dx'], hc_sz_in_age_range['age'],c=cluster[0],  s=50)
        ax.set_title('Cluster 1')
        ax.set_xlabel('dx')
        ax.set_ylabel('age')
        plt.colorbar(scatter)'''
    
    #print(hc_sz_in_age_range['age'].mean())
                                                                        


#### SVM: 
#svm_data_test = pd.read_csv('NCY01_20171112.txt', index_col=[0])

### Using KMeans searching to find the best cluster number:
#cluster2 = KMeans(n_clusters=4, random_state=0).fit(scaled_roi_data)
#print(cluster2.labels_)
#print(cluster2.inertia_)

### Using SVM to classify hc-sz
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV

#hc_sz_data_df = all_feature_df[all_feature_df['dx']==0].append(all_feature_df[all_feature_df['dx']==1])
#hc_sz_feature = hc_sz_data_df.iloc[:,5:].values
#hc_sz_label = hc_sz_data_df.iloc[:,1].values
sc = MinMaxScaler(feature_range=(0,1))
scaled_hc_sz_train = sc.fit_transform(hc_sz_train_data)
scaled_hc_sz_test =sc.transform(hc_sz_test_data)
clusterhc_sz = hierachy_cluster.fit_predict(scaled_hc_sz_train)
clusterhc_sz_df=pd.DataFrame(clusterhc_sz)

print(clusterhc_sz.n_clusters_)
fig = plt.figure()
ax = fig.add_subplot(111)
scatter=ax.scatter(hc_sz_train_df['dx'], hc_sz_train_df['age'],c=clusterhc_sz_df[0],  s=50)
ax.set_title('Cluster 1')
ax.set_xlabel('dx')
ax.set_ylabel('age')
plt.colorbar(scatter)

plt.scatter(hc_sz_train_df['dx'], clusterhc_sz_df[0])
plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(hc_sz_train_data, hc_sz_train_label, test_size=0.2, random_state=0)
sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
scaled_hc_sz_test =sc.transform(hc_sz_test_data)
#### need to use the same scaling---> fit the train data and use the same model to transform test

svm_model = SVC()
parameters = {'kernel':['linear','poly','rbf'], 'C':[0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]}
tuned_model = GridSearchCV(svm_model, parameters).fit(scaled_hc_sz_train, hc_sz_train_label)

print(tuned_model.best_score_, tuned_model.best_params_)


### Using ANN to classify hc-sz
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

ann_model = Sequential()
ann_model.add(Dense(100,activation='relu', kernel_initializer='glorot_uniform', input_dim=237))
ann_model.add(Dense(50,activation='relu', kernel_initializer='glorot_uniform'))
ann_model.add(Dropout(rate=0.4))
ann_model.add(Dense(100,activation='relu', kernel_initializer='glorot_uniform'))
ann_model.add(Dropout(rate=0.4))
ann_model.add(Dense(50,activation='relu', kernel_initializer='glorot_uniform'))
ann_model.add(Dense(50,activation='relu', kernel_initializer='glorot_uniform'))
ann_model.add(Dense(50,activation='relu', kernel_initializer='glorot_uniform'))
ann_model.add(Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform'))
ann_model.summary()
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = ann_model.fit(X_train, y_train, validation_data=(X_test,y_test),batch_size=20, epochs=200)
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


#### Using GridSearch:
''' = KerasClassifier(build_fn=ann_model)
parameters = {'batch_size':[10,20,40], # according to the experience of the tutor
              'epochs':[100,500]}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=5)
grid_result = grid_search.fit(scaled_hc_sz_train, hc_sz_train_label)
best_parameter = grid_result.best_params_
best_accuracy = grid_result.best_scores_'''

score, acc = ann_model.evaluate(X_test, y_test, batch_size=20)
print('Test Score: ', score)
print('Test Accuracy: ', acc)


score, acc = ann_model.evaluate(scaled_hc_sz_test, hc_sz_test_label,batch_size=20)
print('Test Score: ', score)
print('Test Accuracy: ', acc)