# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:00:44 2018

@author: anish
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import DNN
import pickle



with open("parameters.txt","rb") as fp:
    parameters=pickle.load(fp)

for i in range(1,9):
    img=Image.open("test"+str(i)+".jpg")
    img=img.resize((64,64))
    X=np.asarray(img)
    plt.imshow(X)
    X=X.reshape(X.shape[0]*X.shape[1]*X.shape[2],1)
    X=X/255
    result=DNN.predict(parameters,X,0.5)
    print(result)
    input()