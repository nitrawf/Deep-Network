# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:06:21 2018

@author: anish
"""

from os import listdir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import helper_functions as hf

def loaddata():
    mypath="D:\Download\cat-dataset\CAT_00"
    catfiles = [f for f in listdir(mypath)]
    for i in catfiles:
        img=Image.open(mypath+"\\"+i)
        img=img.resize((64,64))
        x=np.asarray(img)
        x=np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],1))
        X_list.append(x)
        Y_list.append(1)
    mypath2="D:\Download\ObjectCategories\BACKGROUND_Google"
    notcatfiles = [g for g in listdir(mypath2)]
    for j in notcatfiles:
        img=Image.open(mypath2+"\\"+j)
        img=img.resize((64,64))
        x=np.asarray(img)
        if x.shape!=(64,64,3):
            continue
        x=np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],1))
        X_list.append(x)
        Y_list.append(0)
        

def forward_prop(X,parameters,layers_dims,Y):
    L=len(parameters)//2
    caches={}
    A=X
    for l in range(1,L+1):
        ln=str(l)
        W=parameters["W"+ln]
        b=parameters["b"+ln]  
        Z=hf.forward_linear(A,W,b)
        caches["Z"+ln]=Z
        caches["W"+ln]=W
        caches["b"+ln]=b
        if l==L+1:
            AL=hf.sigmoid(Z)
        else:
            A=hf.relu(Z)
    return caches,AL


def back_prop(AL,Y,caches):
    dAL=-(np.divide(Y, AL)-np.divide(1-Y,1-AL))
    dZL=Y-AL
    dWL=(1/m)np.dot
    for l in reversed(range(L-1)):
        

#X_list=[]
#Y_list=[]
#loaddata()
#c = list(zip(X_list, Y_list))
#random.shuffle(c)
#X_list, Y_list = zip(*c)
#X=np.array(X_list)
#X=X.reshape((X.shape[0],X.shape[1]))
#X=X.T
#Y=np.array(Y_list)
#Y=Y.reshape((1,Y.shape[0]))
#X=X/255
#X_trainset=X[:,0:2000]
#Y_trainset=Y[:,0:2000]
#X_testset=X[:,2000:]
#Y_testset=Y[:,2000:]        

X=np.random.randn(10,50)
Y=np.random.rand(50,1)
Y=(Y<0.5)

layers_dims=hf.layer_size(3,10,4)    
print(layers_dims)
parameters=hf.initialize(layers_dims)    
print(parameters)
caches,A=forward_prop(X,parameters,layers_dims,Y)
print(A)
    