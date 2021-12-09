# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:13:49 2020

@author: WB
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# importing dataset:
# what engine do?
movies = pd.read_csv('ml-1m/movies.dat',sep='::', header=None, engine='python',encoding='latin-1')
user = pd.read_csv('ml-1m/users.dat',sep='::', header=None, engine='python',encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat',sep='::', header=None, engine='python',encoding='latin-1')

# Preparing train / test:
training_set = pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users / movies
# data attributes: number of user / movies 
numb_usr = int(max(max(training_set[:,0]),max(test_set[:,0])))
numb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Converting data to arrays: row-user / colum-movie
# list of list: for converting to tensor
def convert(data):
    ### Why this function can use numb_movies directly? ---> all variable outside function is gloabal
    new_data=[]
    for id_user in range(1, numb_usr+1):
        id_movie = data[:,1][data[:,0]==id_user]
        id_rating = data[:,2][data[:,0]==id_user]
        ## Using np.array is better: can assign value, but list can't
        ratings =np.zeros(numb_movies)
        ratings[id_movie-1] = id_rating
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)


# Converting data to torch tensor:
# What is a define-by-run framework?
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
#torch.cuda.is_available()---> checking CPU
# Rating: -1 --> haven't seen, 1/2 -->doesn't like
training_set[training_set==0] = -1
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set>=3] = 1

test_set[test_set==0] = -1
test_set[test_set==1] = 0
test_set[test_set==2] = 0
test_set[test_set>=3] = 1

##### Building RBM class:

class RBM():
    # nv: number of visible nodes
    # nh: number of hidden nodes
    def __init__(self, nv, nh):
        # initialize weights according to normal distribution
        # bias for hidden(a): 2D--> first dimention is for batch(?)
        # bias for visible(b)
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)

    
    # Sampling the hidden nodes:
    def sample_h(self, x):
        # x: values from visible nodes
        wx = torch.mm(x, self.W.t())
        # expand_as: insure that the bias applied to each mini batch
        activation = wx + self.a.expand_as(wx)
        # probability of h
        p_h_given_v = torch.sigmoid(activation)
        # Why bernoulli rbm?
        # return probability and bernoulli sample
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    # Sampling the visible nodes:
    def sample_v(self, y):
        # y: values from hidden nodes
        # weight need not to be transposed
        wy = torch.mm(y, self.W)
        # expand_as: insure that the bias applied to each mini batch
        activation = wy + self.b.expand_as(wy)
        # probability of h
        p_v_given_h = torch.sigmoid(activation)
        # Why bernoulli rbm?
        # return probability and bernoulli sample
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    #Contrastive divergence
    # approximating the gradient: using k-step contrastive divergence
    # v0: input-- rating of one user
    # vk: the visible nodes after k round
    # ph0: the probability of hidden node=1  given v0
    # phk: after k
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)
        self.b += torch.sum((v0-vk),0) # 0 is for keeping the dimension consistent
        self.a += torch.sum((ph0-phk),0)


'''
#### RBM : bugs: rbm.train(v0, vk, ph0, phk) 
###       RuntimeError: The size of tensor a (1682) must match the size of tensor b (100) at non-singleton dimension 1
# Creating Object:
# nh is tunnable        
nv =  len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv,nh)

# Train RBF:
n_epoch = 10
for epoch in range(1, n_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(0,numb_usr-batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        # return the first element of the functio return
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s += 1.
    ## print the loss to observe the training: normalized loss
    print('epoch: ' + str(epoch) + ' Loss:' + str(train_loss/s))
    
# Get test set result:
test_loss = 0
s = 0.
for id_user in range(numb_usr-batch_size):
    ### only one user a time
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0])>0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
    test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
    s += 1.
    ## print the loss to observe the training: normalized loss
print('Test Loss:' + str(test_loss/s))    
'''

#### Auto-encoder:
## Overcomplete / Sparse (regulation:restricting hidden units when every input-a bit like anti-dropout) / 
## Denoising Autoencoder: randomly turn some of units to 0  
## Contractive Autoencoder: using penalties to restrict hidden layer
## Stack Autoencoder: two hidden layers--->its possible have better result than DBN 
## Deep Autoencoder: stack of RBN
## Most used: Sparse / Denoising    

## Stack Auto-encoder:
class SAE(nn.Module):
    # Using Inherit
    def __init__(self, ):
        ### Calling the parent's method
        super(SAE,self).__init__()
        # first hidden layer: 20 nerons
        self.fc1 = nn.Linear(numb_movies, 20)
        # Second hidden layer: 10 nerons
        self.fc2 = nn.Linear(20,10)
        # Second hidden layer: 10 nerons
        self.fc3 = nn.Linear(10,20)
        # Output
        self.fc4 = nn.Linear(20, numb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        ## if set different x, can retrieve these encoded pattern(activation not weights)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
sae = SAE()
criterian = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay = 0.5)

## Train the SAE:
# Define number of epochs:
nb_epochs = 200
for epoch in range(1,nb_epochs+1):
    train_loss = 0
    s = 0.
    for id_user in range(numb_usr):
        # adding a new dimension: batch--> required in keras and pytorch
        input_ = Variable(training_set[id_user]).unsqueeze(0)   
        target = input_.clone()
        if torch.sum(target.data > 0) > 0:
            #### Why we don't need to call forward function?
            output = sae(input_)
            target.require_grad = False
            output[target==0] = 0
            loss = criterian(output,target)
            ## adding 1e-10 to avoid n/0 error
            mean_corrector = numb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: ' + str(epoch) + ' Loss:' + str(train_loss/s))

#torch.Tensor is the central class of the package. 
#If you set its attribute .requires_grad as True, it starts to track all operations on it.
#When you finish your computation you can call .backward() and have all the gradients computed automatically. 
#The gradient for this tensor will be accumulated into .grad attribute.


### Test set:
test_loss = 0
s = 0.
for id_user in range(numb_usr):
# adding a new dimension: batch--> required in keras and pytorch
    input_ = Variable(training_set[id_user]).unsqueeze(0)   
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input_)
        target.require_grad = False
        output[target==0] = 0
        loss = criterian(output,target)
        ## adding 1e-10 to avoid n/0 error
        mean_corrector = numb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('Test Loss:' + str(train_loss/s))

## activation function / structure / epoch / learing rate
### parameter and hyper-parameter to Tune    
