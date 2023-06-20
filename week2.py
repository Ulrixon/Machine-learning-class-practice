# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:40:17 2019

@author: User
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

 #file_path = '/home/kian/Dropbox/NTPU/Course/2022Spring/MachineLearning/Week2'
# %%
N = 20
x = np.linspace(0, 2, 500)[:, np.newaxis]
y = 1+1*x
fig1 = plt.figure(figsize=(5,4), dpi=200)
ax = fig1.add_subplot(1,1,1)
ax.plot(x, y)
ax.tick_params(labelsize=16)
ax.set_xlabel(r"$x$",fontsize=16)
ax.set_ylabel(r"$y$",fontsize=16)
ax.set_title(r"$y=1+x$ ($m=1$)",fontsize=16)
plt.show()

# %%
y = 1+1*x-1*(x**2)
fig1 = plt.figure(figsize=(5,4), dpi=200)
ax = fig1.add_subplot(1,1,1)
ax.plot(x, y)
ax.tick_params(labelsize=16)
ax.set_xlabel(r"$x$",fontsize=16)
ax.set_ylabel(r"$y$",fontsize=16)
ax.set_title(r"$y=1+x-x^2$ ($m=2$)",fontsize=16)
plt.show()
# %%
y = 1+1*x-1*(x**2)+0.3*(x**3)
fig1 = plt.figure(figsize=(5,4), dpi=200)
ax = fig1.add_subplot(1,1,1)
ax.plot(x, y)
ax.tick_params(labelsize=16)
ax.set_xlabel(r"$x$",fontsize=16)
ax.set_ylabel(r"$y$",fontsize=16)
ax.set_title(r"$y=1+x-x^2+0.3x^3$ ($m=3$)",fontsize=16)
plt.show()
# %%
# Data generating process
np.random.seed(20)
N =100
x = np.linspace(0, 2, N)[:, np.newaxis]
e = 0.5*np.random.randn(N)[:, np.newaxis]
y = 1+1*x-1*(x**2)+e
plt.scatter(x,y)
# %%
M=10
x0 = np.linspace(0, 2, 500)[:, np.newaxis]
yhat0 = np.zeros((500,M))
power = np.linspace(0,9,M)
X=np.power(x, power)
X0=np.power(x0, power)

    
# %%
L=np.zeros((M,1))
for kk in range(0,M,1):
     Xk= X[:,:kk+1]
     bhat = inv(Xk.T@Xk)@Xk.T@y
     X0k = X0[:,:kk+1]

     yhat = Xk@bhat.flatten()
     yhat0[:,kk]= X0k@bhat.flatten()
     L[kk,0]=sum((y-yhat[:,np.newaxis])**2)/N
# %%
    
c=(1,3,5,9)
for cc in c:
    fig2 = plt.figure(figsize=(5,4), dpi=200)
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(x0, yhat0[:,cc], 'r')
    ax2.plot(x0, 1+1*x0-1*(x0**2),'b')
    ax2.scatter(x,y)
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel(r"$x$",fontsize=16)
    ax2.set_ylabel(r"$y$",fontsize=16)
    dd=L[cc,0]
    ax2.set_title(r"$m=%i$" %cc + ", "+r"$L=%1.2f$" %dd,fontsize=16)
    plt.show()
    #plt.savefig(file_path+'/fig2'+str(cc)+'1.png')
    
# %%# LOOCV
CV1=np.zeros((N,M))
for ii in np.arange(N):
    idx1=(x!=x[ii]).flatten()
    idx2=(x==x[ii]).flatten()
    X1=X[idx1,:]
    y1=y[idx1,:]
    for kk in range(0,M,1):
        X1k=X1[:,:kk+1]
        bhat1 = inv(X1k.T@X1k)@X1k.T@y1
        CV1[ii,kk]=(y[idx2,:]-X[idx2,:kk+1]@bhat1)**2
CVm=CV1.mean(axis=0)

# %%
c=(1,3,5,9)
for cc in c:
    fig3 = plt.figure(figsize=(5,4), dpi=200)
    ax3 = fig3.add_subplot(1,1,1)
    ax3.plot(x0, yhat0[:,cc], 'r')
    ax3.plot(x0, 1+1*x0-1*(x0**2),'b')
    ax3.scatter(x,y)
    ax3.tick_params(labelsize=16)
    ax3.set_xlabel(r"$x$",fontsize=16)
    ax3.set_ylabel(r"$y$",fontsize=16)
    dd=CVm[cc]
    ax3.set_title(r"$m=%i$" %cc + ", "+r"$CV=%1.2f$" %dd,fontsize=16)
    plt.savefig(file_path+'/fig2'+str(cc)+'2.png')
    

    
    

    












