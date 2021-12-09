import scipy.io as spi 
import pandas as pd 
import numpy  as np
import os
import matplotlib.pyplot as plt 
import seaborn as sbn 
import matplotlib.animation as animate
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join
import re
import decimal
import threading
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
import random
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import statistics 
from adspy_shared_utilities import plot_class_regions_for_classifier
from sklearn.linear_model import LinearRegression



linsvm_10_df=pd.read_csv('linsvm_hc_sz_10.csv')
linsvm_25_df=pd.read_csv('linsvm_hc_sz_25.csv')
linsvm_50_df=pd.read_csv('linsvm_hc_sz_50.csv')
linsvm_100_df=pd.read_csv('linsvm_hc_sz_100.csv')
feature_all_df=pd.read_csv('linsvm_feature.csv')
feature_data=feature_all_df.iloc[:,1:]
label_data=feature_all_df.iloc[:,0]

#print(linsvm_50_df)
#print(linsvm_100_df)
#print(feature_all_df)

linsvm_10_pure=linsvm_10_df.iloc[:-1,1:]
linsvm_25_pure=linsvm_25_df.iloc[:-1,1:]
linsvm_50_pure=linsvm_50_df.iloc[:-1,1:]
linsvm_100_pure=linsvm_100_df.iloc[:-1,1:]

#range_list=linsvm_100_df.columns
range_list=[i for i in range(50,210,10)]


def mean_std_list(some_df):
    mean_list=[]
    std_list=[]
    for colm in some_df:
        mean=some_df[colm].mean()
        std=some_df[colm].std()
        mean_list.append(mean)
        std_list.append(std)
    return (mean_list,std_list)


plt.subplot(224)    
ax1=plt.subplot(221)
ax1.errorbar(range_list,mean_std_list(linsvm_50_pure)[0],mean_std_list(linsvm_50_pure)[1], linestyle='-', marker='o',solid_capstyle='butt',label='50',color='r')
ax1.legend()
ax2=plt.subplot(222)
ax2.errorbar(range_list,mean_std_list(linsvm_100_pure)[0],mean_std_list(linsvm_100_pure)[1],linestyle='-', marker='o',solid_capstyle='butt',label='100',color='b')
ax2.legend()
ax3=plt.subplot(223)
ax3.errorbar(range_list,mean_std_list(linsvm_10_pure)[0],mean_std_list(linsvm_10_pure)[1], linestyle='-', marker='o',solid_capstyle='butt',label='10',color='g')
ax3.legend()
ax4=plt.subplot(224)
ax4.errorbar(range_list,mean_std_list(linsvm_25_pure)[0],mean_std_list(linsvm_25_pure)[1], linestyle='-', marker='o',solid_capstyle='butt',label='25',color='c')
ax4.legend()
#plt.show()


#print(label_data)
#print(feature_data)
#feature_pca=PCA(n_components=2).fit(feature_data)
#feature_trans=feature_pca.transform(feature_data)
#plot_class_regions_for_classifier(feature_trans,label_data,[-1,1])


#plotting out-of-sample err

linsvm_out=pd.read_csv('linsvm_out_of_sample_50.csv')
linsvm_out_std_list=[]
linsvm_out_mean_list=[]
#print(linsvm_out.columns)

for i in range(1,len(linsvm_out.columns)):
    linsvm_out_mean_list.append(linsvm_out.iloc[-1,i])
  
for i in range(1,len(linsvm_out.columns)):
    linsvm_out_std_list.append(linsvm_out.iloc[0:-1,i].std())

print(linsvm_out_mean_list)
print(linsvm_out_std_list)
range_list_out=[i for i in range(30,200,10)]

std_array=np.array(linsvm_out_std_list)
range_array=np.array(range_list_out)

linreg=LinearRegression()
model_lin=linreg.fit(range_array.reshape(-1,1),std_array.reshape(-1,1))
print(model_lin.predict(std_array.reshape(-1,1)))
plt.figure()
#plt.errorbar(range_list_out,linsvm_out_mean_list,linsvm_out_std_list,linestyle='-', marker='o',solid_capstyle='butt',label='50',color='b')
plt.scatter(range_array,std_array,label='50')
plt.plot(range_array,model_lin.predict(range_array.reshape(-1,1)),color='r')
plt.legend()
plt.show()