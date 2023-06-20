# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:52:37 2020

@author: Kian
"""
# %%
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
# Simulation
N = 1000

X1 = np.concatenate((-3+np.random.randn(N//2,1),3+np.random.randn(N//2,1)))
X2 = np.concatenate((-3+np.random.randn(N//2,1),3+np.random.randn(N//2,1)))
x = np.concatenate((np.ones((N,1)),X1,X2,X1**2,X2**2),axis=1)
b0 = np.array([1,0.7,-0.1,0.3,-0.1],ndmin=2).T
pri =np.exp(x@b0)/(1+np.exp(x@b0))
y = np.random.binomial(1, pri)
plt.hist(X1)
# %%
def nlikelihood_LR(theta):
    """Logistic regression"""
    global x
    global y
    xb = x@theta[:,np.newaxis]
    lambda_ = np.exp(xb)/(1+np.exp(xb)) # N x 1
    return -(np.sum(y*np.log(lambda_)+(1-y)*np.log(1-lambda_)))

def LR_gradient(theta):
    """Logistic regression"""
    global x
    global y
    xb = x@theta[:,np.newaxis]
    lambda_ = np.exp(xb)/(1+np.exp(xb)) # N x 1
    return -(x.T@(y-lambda_)).flatten()

def LR_hessian(theta):
    """Logistic regression"""
    global x
    global y
    N , k = np.shape(x)
    Hi = np.zeros((k,k,N))
    xb = x@theta[:,np.newaxis]
    lambda_ = np.exp(xb)/(1+np.exp(xb)) # N x 1
    for ii in range(N):
        Hi[:,:,ii] = lambda_[ii,0]*(1-lambda_[ii,0])*x[ii,:].T@x[ii,:]
    return np.sum(Hi,axis=2)

# %%
flag=0
while flag!=1:
    theta0 = np.random.randn(5)/10
    res_NCG = minimize(nlikelihood_LR, theta0, method='Newton-CG',jac=LR_gradient, hess=LR_hessian, options={'maxiter':10000,'disp': True})
    flag = res_NCG.success
print(res_NCG.x)
# %%
flag=0
while flag!=1:
    theta0 = np.random.randn(5)/10
    res_bfgs = minimize(nlikelihood_LR, theta0, method='BFGS',jac=LR_gradient)
    flag = res_bfgs.success
print(res_bfgs.x)
# %%   
flag=0
while flag!=1:
    theta0 = np.random.randn(5)/10
    res_lr = minimize(nlikelihood_LR, theta0, method='Nelder-Mead')
    flag = res_lr.success
    
print(res_lr.x)
# %%
