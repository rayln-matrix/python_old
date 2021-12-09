# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:23:38 2020

@author: WB
"""


import svmutil
import os
import sys
import numpy as np

x1=[1,0]
x2=[0,1]
x3=[0,-1]
x4=[-1,0]
x5=[0,2]
x6=[0,-2]
x7=[-2,0]
set1=[x1,x2,x3,x4,x5,x6,x7]


y1=-1
y2=-1
y3=-1
y4=1
y5=1
y6=1
y7=1
set2=[y1,y2,y3,y4,y5,y6,y7]
set3=[]
for x in set1:
    theta1=x[1]**2-2*x[0]+3
    theta2=x[0]**2-2*x[1]-3
    z=[theta1,theta2]
    #print(z)
    set3.append(z)
    
#print(set3)

#fil=open('test2.txt',mode='w',encoding='utf-8')
#for i in range(len(set1)):
#    for j in range(len(set3)):
##        if i==j:
 #           line=str(set2[i])+' '+'1:'+str(set1[j][0])+' '+'2:'+str(set1[j][1])
 #           print(line)
 #           fil.write(line)
 #           fil.write('\n')#
#
#fil.close

y,x=svmutil.svm_read_problem('test2.txt',return_scipy=True)
m=svmutil.svm_train(y,x,'-s 0 -t 1 -r 1 -g 1 -d 2')
sv=m.get_SV()
svcoef=m.get_sv_coef()
idc=m.get_sv_indices()
#w=[1,2]
b=m.rho

#for svs in sv:
#    for svdic in svs:
#        if svdic!=2:
#            svs[2]=0.0
#    for svdic in svs:    
#        if svdic==2 & len(svs)==1:
#            svs[1]=0.0
#for i in range(len(sv)):
#    for j in range(len(svcoef)):
#        if i==j:
#           w=[w[0]+sv[i][1]*svcoef[j][0],w[1]+sv[i][2]*svcoef[j][0]]
            
#print(w,b[0])
print(sv)
#print(sv[1][1])
print(svcoef)
#print(idc)
sumz=svcoef[0][0]/1+svcoef[1][0]/1+svcoef[2][0]/1+svcoef[3][0]/-1+svcoef[4][0]/-1
print(sumz)
#w=[-1.0*svcoef[0][0],2.0*svcoef[1][0]+(-2.0)*svcoef[2][0]+1.0*svcoef[3][0]+(-1.0)*svcoef[4][0]]
print(b[0])
#for x in w:
#    print('{:.8f}'.format(float(x))) 
    
ya1=svcoef[0][0]
ya2=svcoef[1][0]
ya3=svcoef[2][0]
ya4=svcoef[3][0]
ya5=svcoef[4][0]

phi1=np.array([1,-(2**(0.5)),0,0,2,0])
phi2=np.array([1,0,2*(2**(0.5)),0,0,4])
phi3=np.array([1,0,-2*(2**(0.5)),0,0,4])
phi4=np.array([1,0,(2**(0.5)),0,0,1])
phi5=np.array([1,0,-1*(2**(0.5)),0,0,1])

w=ya1*phi1+ya2*phi2+ya3*phi3+ya4*phi4+ya5*phi5
print(w.round(1))