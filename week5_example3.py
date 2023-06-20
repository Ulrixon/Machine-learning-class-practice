# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:18:47 2020

@author: Kian
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from numpy.linalg import inv
# Simulation
N = 20
np.random.seed(20)
X1 = np.concatenate((-3+np.random.randn(N//2,1),3+np.random.randn(N//2,1)))
X2 = np.concatenate((-3+np.random.randn(N//2,1),3+np.random.randn(N//2,1)))
X = np.concatenate((np.ones((N,1)),X1,X2,X1**2,X2**2),axis=1)
#X = np.concatenate((np.ones((N,1)),X1,X2),axis=1)
b0 = np.array([1,0.7,-0.3,0.3,-0.3],ndmin=2).T
#b0 = np.array([1,0.7,-0.1],ndmin=2).T
pri =1/(1+np.exp(-X@b0))
y = np.random.binomial(1, pri)


toymodel_lr = LogisticRegression(fit_intercept=False).fit(X,y)
bhat = toymodel_lr.coef_

from matplotlib. colors import ListedColormap
markers = ('s','x')
colors = ('red','blue')
cmap = ListedColormap(('pink','lightblue'))

x1min , x1max = X[:,1].min()-1, X[:,1].max()+1
x2min , x2max = X[:,2].min()-1, X[:,2].max()+1
xx1 , xx2 = np.meshgrid(np.arange(x1min,x1max,0.02),np.arange(x2min,x2max,0.02))
N1,N2 = np.shape(xx1)
xb1 = np.concatenate((np.reshape(xx1,[N1*N2,1]),np.reshape(xx2,[N1*N2,1]),np.reshape(xx1,[N1*N2,1])**2,np.reshape(xx2,[N1*N2,1])**2),axis=1)@bhat[:,1:5].T+bhat[0,0]
#xb1 = np.concatenate((np.reshape(xx1,[N1*N2,1]),np.reshape(xx2,[N1*N2,1])),axis=1)@bhat[:,1:3].T+bhat[0,0]
z = np.exp(xb1)/(1+np.exp(xb1))
zhat= np.where(z>0.5,1,0)
fig1 = plt.figure(figsize=(5,4), dpi=200)
plt.contourf(xx1,xx2,z.reshape(xx1.shape),aplha=0.3,cmap=cmap)

for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x =X[y.ravel()==cl,1],y=X[y.ravel()==cl,2],alpha=0.8,c=colors[idx],marker=markers[idx],label=cl,edgecolor='black')
