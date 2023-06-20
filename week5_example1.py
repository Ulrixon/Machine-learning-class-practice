# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:20:02 2020

@author: Kian
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
# %%
file_path='~/downloads'
data = pd.read_csv(file_path+'/creditcard.csv', header=0)
# %%
data0 = data[data['Class']==0]
data1 = data[data['Class']==1]

data0_train, data0_test = train_test_split(data0, train_size=300)
data1_train, data1_test = train_test_split(data1, train_size=300)
data0_test = data0_test.sample(192)

y_train = pd.concat([data0_train['Class'], data1_train['Class']], axis=0)
x_train = pd.concat([data0_train.loc[:,'V1':'V28'], data1_train.loc[:,'V1':'V28']], axis=0)

y_test = pd.concat([data0_test['Class'], data1_test['Class']], axis=0)
x_test = pd.concat([data0_test.loc[:,'V1':'V28'], data1_test.loc[:,'V1':'V28']], axis=0)

model_lr = LogisticRegression(fit_intercept=False).fit(x_train,y_train)
parameters = model_lr.coef_
yhat = np.exp(x_test@parameters.T)/(1+np.exp(x_test@parameters.T))
yhat[yhat>=0.5]=1
yhat[yhat<0.5]=0

print(confusion_matrix(y_test, yhat))
# %%
n=tn, fp, fn, tp
for i in range(4):
    
    tn, fp, fn, tp = confusion_matrix(y_test, yhat)
print(precision_score(y_test, yhat))


# %%
from numpy.linalg import inv

bhat0 = np.zeros([28,1])+np.random.randn(28,1)/100
eta = 0.001
criterion = 0.0001
max_iters = 500
ii = 0
dd = 999
logL = np.zeros([max_iters+1,1])
logL[0,0] = -999
y_train = pd.concat([data0_train['Class'], data1_train['Class']], axis=0)

y_train=y_train[:,np.newaxis]

while dd > criterion and ii < max_iters:
    xb = x_train@bhat0[:,ii:ii+1]
    lambda_ = np.exp(xb)/(1+np.exp(xb)) # N x 1
    g = x_train.T@(y_train-lambda_)
    bhat0 = np.append(bhat0, bhat0[:,ii:ii+1]+eta*g, axis=1)
    logL[ii+1,0] = np.sum(y_train*np.log(lambda_)+(1-y_train)*np.log(1-lambda_))
    dd = abs((logL[ii+1,0]-logL[ii,0])/logL[ii,0])
    ii += 1
















# %%
