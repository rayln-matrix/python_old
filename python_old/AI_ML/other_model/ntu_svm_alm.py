# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:16:19 2020

@author: WB
"""
import svmutil
import numpy as np
import random

y,x=svmutil.svm_read_problem('amltrain0.txt',return_scipy=True)
m=svmutil.svm_train(y,x,'-s 0 -t 2 -g 100 -c 0.01')
svmutil.svm_save_model('alm.model', m)
#print(m.sv_coef)
#sv=m.SV
#print(type(sv))
#svcoef=m.sv_coef
#print(type(svcoef))
w=[0,0]
sv=m.get_SV()
svcoef=m.get_sv_coef()
#print(svcoef[2][0])
#print(type(svcoef[2]))
#print(sv[0])
#print(sv[0][1])
#w=np.dot(sv,svcoef)
for i in range(len(sv)):
    for j in range(len(svcoef)):
        if i==j:
            w=[w[0]+sv[i][1]*svcoef[j][0],w[1]+sv[i][2]*svcoef[j][0]]
#svcoef:yn*alpha(n)
            
#print(sv)
#print(svcoef)
#print(w)
print((w[0]**2+w[1]**2)**(0.5))

y,x=svmutil.svm_read_problem('amltest0.txt',return_scipy=True)
p_labs, p_acc, p_vals=svmutil.svm_predict(y, x, m)
print(p_acc)

#for numb in range(len(svcoef)):
#    print(svcoef[numb][0])    
    
    
alphasum=0.0
#print(y[0])
#print(svcoef[0])
fil1=open('aml_alpha.txt',mode='w',encoding='utf-8')
for k in range(len(y)):
    for h in range(len(svcoef)):
        if k==h:
            alphan=abs(round(svcoef[h][0],2))#/round(y[k])
            fil1.write(str(alphan))
            fil1.write('\n')
            fil1.write(str(alphasum))
            fil1.write('\n')
            alphasum=round(alphan,2)+round(alphasum,2)
            #print(alphasum)


print(alphasum)
            
            
#-----------------random sampling-------------------------------------
            
for g in [1,10,100,1000,10000]:
    y,x=svmutil.svm_read_problem('amltrain0.txt',return_scipy=True)
    m=svmutil.svm_train(y,x,'-s 0 -t 2 -g %d -c 0.01'%g)
    #svmutil.svm_save_model('alm.model', m0) 
    y,x=svmutil.svm_read_problem('amltest0.txt',return_scipy=True)
    p_labs, p_acc, p_vals=svmutil.svm_predict(y, x, m)
    print(p_acc)