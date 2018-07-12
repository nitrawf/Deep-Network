
# coding: utf-8

# In[4]:


from os import listdir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle



# In[5]:


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
    dz=(z>0)
    return dz


# In[6]:


def layer_size(l,X,n_h):
    layers=[]
    layers.append(X.shape[0])
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


# In[24]:


def loaddata():
    counter=1
    notcatfiles_list=["airplanes","BACKGROUND_Google","camera","cup","dolphin","headphone","mayfly","revolver","sunflower","wheelchair"]
    for i in notcatfiles_list:
        mypath2="D:\Download\ObjectCategories\\"+i
        notcatfiles = [g for g in listdir(mypath2)]
        print(counter)
        counter+=1
        for j in notcatfiles:
            img=Image.open(mypath2+"\\"+j)
            img=img.resize((64,64))
            x=np.asarray(img)
            if x.shape!=(64,64,3):
                continue
            x=np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],1))
            X_list.append(x)
            Y_list.append(0)    
    mypath="D:\Download\cat-dataset\CAT_00"
    catfiles = [f for f in listdir(mypath)]
    for i in catfiles:
        img=Image.open(mypath+"\\"+i)
        img=img.resize((64,64))
        x=np.asarray(img)
        x=np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],1))
        X_list.append(x)
        Y_list.append(1)
    print(counter)
    counter+=1    
    mypath="D:\Download\cat-dataset\CAT_01"
    catfiles = [f for f in listdir(mypath)]
    for i in catfiles:
        img=Image.open(mypath+"\\"+i)
        img=img.resize((64,64))
        x=np.asarray(img)
        x=np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2],1))
        X_list.append(x)
        Y_list.append(1)
    print(counter)
    counter+=1




# In[8]:


def forward_prop(X,parameters):
    L=len(parameters)//2
    caches={}
    A=X
    for l in range(1,L+1):
        ln=str(l)
        W=parameters["W"+ln]
        b=parameters["b"+ln]  
        Z=forward_linear(A,W,b)
        caches["Z"+ln]=Z
        caches["W"+ln]=W
        caches["b"+ln]=b
        if l==L:
            A=sigmoid(Z)
        else:
            A=relu(Z)
        caches["A"+ln]=A
    caches["A"+str(0)]=X
    return caches,A


# In[20]:


def back_prop(Y,caches,L):
    m=Y.shape[1]
    AL=caches["A"+str(L)] 
    A_prev=caches["A"+str(L-1)]
    grads={}
    grads["dZ"+str(L)] = AL-Y
    grads["dW"+str(L)] = (1/m)*np.dot(grads["dZ"+str(L)],A_prev.T)
    grads["db"+str(L)] = (1/m)*np.sum(grads["dZ"+str(L)],axis=1,keepdims=True)
    for l in range(L-1,0,-1):
#        print(l)
        grads["dZ"+str(l)] = np.dot(caches["W"+str(l+1)].T,grads["dZ"+str(l+1)])*relu_back(caches["Z"+str(l)])
        grads["dW"+str(l)] = (1/m)*np.dot(grads["dZ"+str(l)],caches["A"+str(l-1)].T)
        grads["db"+str(l)] = (1/m)*np.sum(grads["dZ"+str(l)],axis=1,keepdims=True)
    return grads


# In[19]:


def update_parameters(parameters,grads,learning_rate):
    L=len(parameters)//2
    for l in range(1,L+1):
        parameters["W"+str(l)]-=grads["dW"+str(l)]*learning_rate
        parameters["b"+str(l)]-=grads["db"+str(l)]*learning_rate
    return parameters


# In[11]:


def predict(parameters, X,threshold):  
    caches,AL = forward_prop(X, parameters)
    predictions=(AL>threshold)       
    return predictions


# In[22]:


def deepmodel(X,Y,learning_rate,iterations,layers_dims):
    costs=[]
    parameters=initialize(layers_dims)
    L=len(parameters)//2
    print("L="+str(L))
    m=Y.shape[1]
    print("m="+str(m))
    for i in range(iterations):
        caches,AL=forward_prop(X,parameters)
#        for key,value in caches.items():
#            print(key)
#            print(value.shape)
        current_cost=cost(AL,Y,m)
        if i%100==0:
            costs.append(current_cost)
            print("Cost of iteration "+str(i)+"= "+str(current_cost))
        grads=back_prop(Y,caches,L)
#        for key,value in grads.items():
#            print(key)
#            print(value.shape)
        parameters=update_parameters(parameters,grads,learning_rate)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()    
    return parameters


# In[25]:

if __name__ == "__main__":
    X_list=[]
    Y_list=[]
    loaddata()
    c = list(zip(X_list, Y_list))
    random.shuffle(c)
    X_list, Y_list = zip(*c)
    X=np.array(X_list)
    X=X.reshape((X.shape[0],X.shape[1]))
    X=X.T
    Y=np.array(Y_list)
    Y=Y.reshape((1,Y.shape[0]))
    X=X/255
    X_trainset=X[:,0:int(X.shape[1]*0.9)]
    Y_trainset=Y[:,0:int(X.shape[1]*0.9)]
    X_testset=X[:,int(X.shape[1]*0.9):]
    Y_testset=Y[:,int(X.shape[1]*0.9):] 
    
    
    # In[34]:
    
    
    #layers_dims=layer_size(5,X,5)
    layers_dims=[X.shape[0],4,4,4,4,1]
    parameters=deepmodel(X_trainset,Y_trainset,0.005,1500,layers_dims)
    t=0.5
    with open("parameters.txt","wb") as fp:
        pickle.dump(parameters,fp)
    
    # In[35]:
    
    
    predictions_train = predict(parameters, X_trainset,t)
    print ('Accuracy for training set with threshold %f: %d' %(t,float((np.dot(Y_trainset,predictions_train.T) + np.dot(1-Y_trainset,1-predictions_train.T))/float(Y_trainset.size)*100)))
    
    predictions_test = predict(parameters, X_testset,t)
    print ('Accuracy for test set: %d' % float((np.dot(Y_testset,predictions_test.T) + np.dot(1-Y_testset,1-predictions_test.T))/float(Y_testset.size)*100))

