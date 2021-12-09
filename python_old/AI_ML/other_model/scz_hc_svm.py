# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:01:37 2020

@author: WB
"""


import svmutil
import os
import sys
import numpy as np
import random
from random import shuffle
import scipy


#讀取格式化的檔案
label_data,feature_data=svmutil.svm_read_problem('feature_all.txt',return_scipy=True)
#label_data: label 檔案
#feature_data: 所有的資料，400筆每筆為長度75*76的資料~由n個d=75的Vector串聯(10的準確率不高) 

#print(type(label_data))
#print(type(feature_data))
#print(len(label_data))
#print(label_data)
hc_label_index_list=[]
sch_label_index_list=[]
for i in range(len(label_data)):
    if label_data[i]==1:
        sch_label_index_list.append(i)
    if label_data[i]==-1:
        hc_label_index_list.append(i)
#print(feature_data[399])
#print(hc_label_index_list)
#print(sch_label_index_list)

#把HC/SCH兩團取出來:---->直接用原本資料抽取的話可能會兩個Group抽到不同數量
hc_feature=feature_data[hc_label_index_list]
sch_feature=feature_data[sch_label_index_list]
#print(hc_feature)
#print(sch_feature)
#print(hc_feature.shape)
#print(sch_feature.shape)

#可以用scipy.sparse.vstack來堆疊兩個feature類別
#hc_label_data=label_data[hc_label_index_list]--->label array無法直接這樣取，以下用np.array直接指派
#sch_label_data=label_data[sch_label_index_list]
#feature_data_sub=scipy.sparse.vstack([hc_feature,sch_feature])

#隨機取樣index:range(50-200,10)
for size in range(50,210,10):
    sample_index_list=[idx for idx in range(200)]
    sample_size=size
    random_sample_index_list=random.sample(sample_index_list,sample_size)
    #用Index取出特定長度的group然後再合併
    hc_sample_feature=hc_feature[random_sample_index_list]
    sch_sample_feature=sch_feature[random_sample_index_list]
    sample_feature_data=scipy.sparse.vstack([hc_sample_feature,sch_sample_feature])
    #Label資料直接指派然後堆疊 
    sch_label_data_sub=np.array([1]*sample_size)
    hc_label_data_sub=np.array([-1]*sample_size)
    sample_label_data=np.concatenate((hc_label_data_sub,sch_label_data_sub))

    #用洗牌方式打亂原本的feature_data    
    indice=np.arange(sample_label_data.shape[0])
    shuffle(indice)
    random_sample_feature_data=sample_feature_data[indice]
    random_sample_label_data=sample_label_data[indice]
    #gamma_range_list=[g for g in range(1,100)]
    range_array=np.arange(0.001,0.10,0.001) 
    range_list=[round(numb,3) for numb in range_array]
    range_list_int=[i for i in range(1,50)]
    itera=0
    #切割資料為training/test，切割方式是簡單的採用前20%為測試(5-fold)，之後可以用洗牌的方式重新採用
    #f=open('result_libsvm_python_60.txt','w')
    #f.write('Using libsvm_python for svm classification and 5-fold validation'+'\n')
    #f.write('\n')
    acc_list=[]
    while itera<5:
        test_size=int(sample_size/5)
        index_number_list=[n for n in range(sample_size)]
        #print(index_number_list)
        test_index=random.sample(index_number_list,test_size)
        #print(test_index)
        train_index=list(set(index_number_list)-set(test_index))
        #print(train_index)
        training_data=random_sample_feature_data[train_index]
        traing_label=random_sample_label_data[train_index]
        test_data=random_sample_feature_data[test_index]
        test_label=random_sample_label_data[test_index]
        for c in range_list:
        #c=0.1
            model_1=svmutil.svm_train(traing_label,training_data,'-s 0 -t 0 -c %f'%c)
            p_labs, p_acc, p_vals=svmutil.svm_predict(test_label, test_data, model_1)
            #f.write('Kernel=linear, C=%.2f'%c+'\n')
            #f.write('Accuracy='+str(p_acc[0])+'%'+'\n')
            acc_list.append(p_acc[0])
        itera += 1
        #f.write('\n')
    sum=0
    for mean in acc_list:
        sum=sum+mean

    print(range_list)
    mean_of_mean=sum/len(acc_list)    
#f.write('Mean accuracy of all itertions:'+str(mean_of_mean)+'%')
#f.close()
#print(p_labs)--->model輸出的label
#print(len(p_vals))#-->??is the probability of getting one of your labels for each data-point? 
#print(acc_list)