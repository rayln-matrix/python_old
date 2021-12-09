import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import random
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import statistics 
from adspy_shared_utilities import plot_class_regions_for_classifier
from sklearn.neural_network import MLPClassifier


#讀取data
feature_label_df=pd.read_csv('linsvm_feature_label.csv')
#print(feature_label_df)

#尚未抽樣的Data集，資料型態是Dataframe
feature_data=feature_label_df.iloc[:,1:]
label_data=feature_label_df.iloc[:,0]
#HC:-1,SZ:1--->
hc_index=label_data[label_data==-1].index.tolist()
sz_index=label_data[label_data==1].index.tolist()
#print(hc_index)
#print(sz_index)

best_para_dict=dict()
para_list=[]
acc_list=[]
c_range=np.linspace(0.1,10,5)
para_search={'C':c_range}

for sample_size in range(30,50,10):
    #loops=0
    #while loops < 2:
    hc_random_index=random.sample(hc_index,sample_size)
    sz_random_index=random.sample(sz_index,sample_size)
    mixed_random_index=set(hc_random_index).union(set(sz_random_index))
    feature_data_random=feature_data.iloc[list(mixed_random_index)]
    label_data_random=label_data.iloc[list(mixed_random_index)]
    X_train,X_test,y_train, y_test=train_test_split(feature_data_random,label_data_random,test_size=0.2,random_state=0)
    svc_model=svm.SVC()#.fit(X_train,y_train)
    #c_range=np.linspace(0.1,10,20)
    #para_search={'C':c_range}
    grid_search_svc=GridSearchCV(svc_model,para_search).fit(X_train,y_train)
    para_list.append(grid_search_svc.best_params_)
    acc_list.append(grid_search_svc.cv_results_['mean_test_score'].mean())
    best_para_dict[sample_size]=para_list
    best_para_dict['mean score of %d'%sample_size]=acc_list
    #loops += 1 


best_para_key=list(best_para_dict.keys())
best_para_value=list(best_para_dict.values())

print(best_para_key)
print(best_para_dict)
best_c=[]
best_mean=[]
best_c.append(best_para_value[0][0]['C'])
best_mean.append(best_para_value[1][0])
print(best_c)
print(best_mean)

best_para_df=pd.DataFrame(best_para_dict)
print(best_para_df)
#best_para_df.to_csv('best_para_svm.csv')
#print(best_para_dict)
X_train,X_test,y_train, y_test=train_test_split(feature_data,label_data,test_size=0.2,random_state=0)
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.iloc[1].shape)
#svc_model=svm.SVC().fit(X_train,y_train)
#for i in range(len(X_test)):
#    #print(X_test[i])
#    print(svc_model.predict(X_test.iloc[i].T))

#print(svc_model.predict(X_test))
#print(svc_model.score(X_test,y_test))


#c_range=np.linspace(0.1,10,100) 
#print(c_range)
#1st trial: 0.01-10,100
#2nd trial: 0.1-10,100

#print(c_range)
# para_search={'C':c_range}
#c_range=[0.1,0.5,1.0,1.5,2.5,5.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0]
c_range=[i for i in range(1,201)]
print(len(c_range))
para_search={'C':c_range}
grid_search_svc=GridSearchCV(svc_model,para_search).fit(X_train,y_train)
print(grid_search_svc.cv_results_['mean_test_score'])
# print(grid_search_svc.cv_results_)
print(grid_search_svc.best_params_)
print(grid_search_svc.best_score_)
# print(len(grid_search_svc.cv_results_['mean_test_score']))


# model1=MLPClassifier(random_state=0, max_iter=300)
# parameters={'learning_rate':[0.1,0.2,0.3,0.4,0.5],'solver':['lbfgs', 'sgd', 'adam']}
# model1_result=GridSearchCV(model1,parameters)
# means = model1_result.cv_results_['mean_test_score']
# stds = model1_result.cv_results_['std_test_score']
# print(means)
