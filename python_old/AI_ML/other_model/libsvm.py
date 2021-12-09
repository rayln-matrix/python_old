import pandas as pd 
import numpy as np 
import os 
import re
import sys


#with open('aml_feature_train.txt',mode='r+',encoding='utf-8') as fil1:
#    for lin in fil1:
#        print(lin)
#with open('aml_feature_test.txt',mode='r+',encoding='utf-8') as fil2:

lisvm_train=pd.read_csv('aml_feature_train.txt',header=None)
#print(lisvm_train)
lbtrans=pd.DataFrame(columns=[0,1,2])
i=0
for line in lisvm_train[0]:
    s1=line.split()
    fit1=s1[1]
    fit2=s1[2]
    id=s1[0]
    #print(type(fit1))
    #print(type(s1[1]))
    s11='1:'+fit1
    s12='2:'+fit2
    #line.replace(line[1],s11)
    #print(type(float(id)))
    if float(id)==8.0:
        id=1
    else:
        id=-1

    # 0的為1類，其他為-1類
    lbtrans.loc[i,0]=id
    lbtrans.loc[i,1]=s11
    lbtrans.loc[i,2]=s12
    i+=1
#print(lisvm_train)
#print(lbtrans)

lisvm_test=pd.read_csv('aml_feature_test.txt',header=None)
#print(lisvm_train)
lbtesttrans=pd.DataFrame(columns=[0,1,2])
k=0
for line in lisvm_test[0]:
    s1=line.split()
    fit1=s1[1]
    fit2=s1[2]
    id=s1[0]
    #print(type(fit1))
    #print(type(s1[1]))
    s11='1:'+fit1
    s12='2:'+fit2
    #line.replace(line[1],s11)
    #print(id)
    if float(id)==8.0:
        id=1
    else:
        id=-1
        
    lbtesttrans.loc[k,0]=id
    lbtesttrans.loc[k,1]=s11
    lbtesttrans.loc[k,2]=s12
    k+=1

lbtrans.to_csv('amltrain8.txt',index=False,header=None,sep=' ')
lbtesttrans.to_csv('amltest8.txt',index=False,header=None,sep=' ')