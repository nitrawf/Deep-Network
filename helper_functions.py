# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:00:21 2018

@author: anish
"""

import numpy as np

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

def relu(z):
    r=np.maximum(0,z)
    return r

def sigmoid_back(z):
    dz=z(1-z)
    return dz

def relu_back(z):
    if z>0:
        dz=1
    else:
        dz=0
    return dz

def layer_size(l,X,n_h):
    layers=[]
    layers.append(X)
    for i in range(1,l-1):
        layers.append(n_h)
    layers.append(1)
    return layers

def initialize(layers_dims):
    parameters={}
    for l in range(1,len(layers_dims)):
        parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*(np.sqrt(2/layers_dims[l-1]))
        parameters["b"+str(l)]=np.zeros((layers_dims[l],1))
    return parameters

def forward_linear(A_prev,W,b):
    Z=np.dot(W,A_prev)+b
    return Z

def cost(A,Y,m):
    c=(-1/m)*np.sum(np.dot(Y,np.log(A).T)+np.dot(1-Y,np.log(1-A).T))
    return c
