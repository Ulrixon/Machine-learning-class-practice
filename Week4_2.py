#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:17:26 2022

@author: kian
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

np.random.seed(20)
N0 = [20,30,40,50]

M=10
power = np.linspace(0,9,M)
L = np.zeros((M,len(N0)))
EV = np.zeros((M,len(N0)))
CVm = np.zeros((M,len(N0)))
fig0 , ax = plt.subplots(2,2,figsize=(16,12),dpi=100,sharey=True)
for nn,N in enumerate(N0):
    # data generating process
    x = np.linspace(0, 2, N)[:, np.newaxis]
    e = 0.5*np.random.randn(N,1)
    y = 1+1*x-1*(x**2)+e

    X=np.power(x, power)
    
    for kk in range(0,M,1):
         Xk= X[:,:kk+1]
         bhat = inv(Xk.T@Xk)@Xk.T@y
         yhat = Xk@bhat
         L[kk,nn]=sum((y-yhat)**2)/N
        
    # LOOCV
    CV1=np.zeros((M,N))
    for ii in np.arange(N):
        idx1=(x!=x[ii]).flatten()
        idx2=(x==x[ii]).flatten()
        X1=X[idx1,:]
        y1=y[idx1,:]
        for kk in range(0,M,1):
            X1k=X1[:,:kk+1]
            bhat1 = inv(X1k.T@X1k)@X1k.T@y1
            CV1[kk,ii]=(y[idx2,:]-X[idx2,:kk+1]@bhat1)**2
    CVm[:,nn]=CV1.mean(axis=1)
    EV=CVm-L
    i2 = nn%2
    i1= nn//2
    ax[i1,i2].plot(L[:,nn], label=r'Empirical Loss')
    ax[i1,i2].plot(CVm[:,nn], label=r'LOOCV')
    ax[i1,i2].plot(EV[:,nn], label=r'Estimation Variance')
    ax[i1,i2].set_xlabel('model complexity')
    ax[i1,i2].legend(loc='upper right')
    ax[i1,i2].set_ylim(0,4)
    ax[i1,i2].set_title(r'$N$='+str(N))

