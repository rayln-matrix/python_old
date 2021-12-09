# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from cvxopt import matrix, solvers
from cvxopt.blas import dot

x1=[1,0]
x2=[0,1]
x3=[0,-1]
x4=[-1,0]
x5=[0,2]
x6=[0,-2]
x7=[-2,0]
set1=[x1,x2,x3,x4,x5,x6,x7]
#print(x1)

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

d=matrix([0,0])
I=np.identity(2)


#print(d)    
Id=matrix(I)
#print(set3)
alist=[]
#for y in set2:
    #for z in set3:
        #z1=matrix(z)
        #print(z1)
        #a=y*matrix([1,z1])
        #print(a)
        #alist.append(a
for i in range(len(set2)):
    for j in range(len(set3)):
        if i==j:
            z1=matrix(set3[j])
            a=set2[i]*matrix([1,z1])
            alist.append(a)

       
#print(alist)
Q=matrix([[0.0,d],[d.T,Id]])    
#print(Q)
q=matrix([0,0,0],tc='d')
#print(q)
glist=[]
for k in range(len(alist)):
    g=-1*matrix(alist[k])
    g1=g.T
    glist.append(g1)
    
#print(glist)
#gnp=np.array(glist)
#print(gnp)
G=matrix(glist,(7,3),tc='d')
#print(G)

#G=-1*matrix(alist,tc='d')
h=matrix([1,1,1,1,1,1,1],tc='d')
h=h*-1
#print(h)
#sol=solvers.qp(Q,q,G,h)
#print(sol['x'])
#opt=sol['x']
#for x in opt:
    #print('{:.8f}'.format(float(x)))
#cvxopt的matrix和np.array不同
#q=2*matrix([[1,2],[3,4]])
#a=np.array([[1,2],[3,4]])
##df=pd.Series([1,2,3,4,5,6,7])
#print(df)
#print(a*2)
#print(q)

#t=[1,2,2,2,1,1,1,1,13,45,5,5,5,6,6,6,1,1,1,1,1,2,2,2,2,33,33,33,33]

#以下是thinkstat
#hist={}
#for x in t:
    #get(key,value)=get value by key, if key doesn't exist,then return the value in get
    #hist[x]=hist.get(x,0)+1
    
#print(hist)

#ji={'x':1,'t':2}
#print(ji.get('z',0))
    
 ##########################################################
    
#k=(1+x.T*x)**2
Id2=np.identity(7)
I2=matrix(Id2)
#print(I2)
    
yn=matrix(set2,tc='d')
#print(yn)

G1=matrix([yn.T,I2],tc='d')
G1=G1*-1

hlist=[]
for a in range(8):
    hlist.append(0)
#print(hlist)
h1=matrix(hlist,tc='d')
#print(G1)
#print(h)

# 計算 x.T*x
ksum=0
'''for n in range(len(set1)):
    for m in range(len(set1)):
        if n==m:
            x1=matrix(set1[n],tc='d')
            x2=matrix(set1[m],tc='d')
            knm=dot(x1,x2)
            ksum=ksum+knm
            #print(knm)'''

#print(ksum)#x.T*x=ksum

#kernel=(1+ksum)**2
#print(kernel)

# 計算Q=ynymk(xn,xm)
qlist=[]
#print(set1)
for g in range(len(yn)):
    #print(g)
    for l in range(len(yn)):
        #print(h)
        x1=matrix(set1[g],tc='d')
        x2=matrix(set1[l],tc='d')
        kernel=(1+dot(x1,x2))**2
        #print(kernel)
        qnm=yn[g]*yn[l]*kernel
        qlist.append(qnm)
 
#print(len(qlist))       
Qk=matrix(qlist,(7,7),tc='d')

plist=[]
for i in range(7):
    plist.append(-1)

qn=matrix(plist,tc='d')

#print(Qk)
#print(qn)
#print(G1)
#print(h1)
sol=solvers.qp(Qk,qn,G1,h1)
opt=sol['x']

proxsum=0
#for x in opt:
    #print('{:.8f}'.format(float(x))) 

for s in opt:
    proxsum=proxsum+s
    
#print(proxsum)

#phi(x)=(1,2*x1,2*x2,x1**2,x1*x2,x2*x1,x2**2)-->kernal
#wsum=matrix([0,0],tc='d')

svx=set1[1]
svy=set2[1]
#print(svx,svy)

def phi(lst):
    l=matrix(lst)
    ans=[]
    ans.append(1)
    ans.append(2*l[0])
    ans.append(2*l[1])
    ans.append(l[0]**2)
    ans.append(l[0]*l[1])
    ans.append(l[1]*l[0])
    ans.append(l[1]**2)
    return ans

w=np.array([0,0,0,0,0,0,0])
for x in range(len(set1)):
    trfx=phi(set1[x])
    trfx=np.array(trfx)
    for y in range(len(yn)):
        for a in range(len(opt)):
            if x==y & x==a:
                alpha=opt[a]
                ym=yn[y]
                #print(trfx)
                sm=alpha*ym*trfx
                w=w+sm

for x in w:
    print('{:.8f}'.format(float(x))) 
#print(w)        
zx=np.array(phi(set1[1]))
#print(type(zx))
#print(type(w))
wz=np.dot(w,zx)
#print(wz)
b=yn[1]-wz
print(b)


            