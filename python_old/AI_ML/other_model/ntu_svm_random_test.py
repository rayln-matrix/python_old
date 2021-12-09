# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:02:58 2020

@author: WB
"""


import svmutil
import numpy as np
import random


y,x=svmutil.svm_read_problem('amltest0.txt',return_scipy=True)
#print(len(y))
#print(x)
lenofldata=[i for i in range(2007)]
setoflen=set(lenofldata)
n_1=0
n_10=0
n_100=0
n_1000=0
n_10000=0
sampletime=0
while sampletime<100:
    random_sample_list=random.sample(range(2007),1000)
    #print(y[random_sample_list])
    setofsampleindx=set(random_sample_list)
    diff=list(setoflen-setofsampleindx)
    #print(diff)
    p=1.0
    best=0
    for g in [1,10,100,1000,10000]:
        m=svmutil.svm_train(y[random_sample_list],x[random_sample_list],'-s 0 -t 2 -g %d -c 0.1'%g)
        yt=y[diff]
        xt=x[diff]
        p_labs, p_acc, p_vals=svmutil.svm_predict(yt, xt, m)
        if p_acc[1]<p:
            p=p_acc[1]
            best=g
    if best==1:
        n_1+=1
    if best==10:
        n_10+=1
    if best==100:
        n_100+=1
    if best==1000:
        n_1000+=1
    if best==10000:
        n_10000+=1
    sampletime+=1
    


#print(p)
#print(best)
print(n_1,n_10,n_100,n_1000,n_10000)
    