# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:15:33 2021

@author: ntpu_metrics 美辰
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
%matplotlib inline

yourpath = 'C:/Users/ntpu_metrics/Documents/Python Scripts/Datasets_ISLR/'

#define a function to draw a plot of an SVM
def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

#SVM
np.random.seed(8)
X = np.random.randn(200,2)
X[:100] = X[:100] +2
X[101:150] = X[101:150] -2
y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)]) #0-149=-1, 150-199=1

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2)

#plot raw data
plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')

#try different C and gamma using grid search
tuned_parameters = [{'C': [0.01, 0.1, 1, 10, 100],'gamma': [0.5, 1,2,3,4]}]
clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X_train, y_train)
clf.best_params_ #{'C': 10, 'gamma': 0.5}

svm = SVC(C=10, kernel='rbf', gamma=0.5)
svm.fit(X_train, y_train)
plot_svc(svm, X_test, y_test)
# or using: plot_svc(clf.best_estimator_, X_test, y_test)
print(confusion_matrix(y_test, clf.best_estimator_.predict(X_test)))
print(clf.best_estimator_.score(X_test, y_test))


#Application 
#import Gene Expression Data
X_train = pd.read_csv(str(yourpath)+'Khan_xtrain.csv').drop('Unnamed: 0', axis=1)
y_train = pd.read_csv(str(yourpath)+'Khan_ytrain.csv').drop('Unnamed: 0', axis=1).values.ravel()
X_test = pd.read_csv(str(yourpath)+'Khan_xtest.csv').drop('Unnamed: 0', axis=1)
y_test = pd.read_csv(str(yourpath)+'Khan_ytest.csv').drop('Unnamed: 0', axis=1).values.ravel()

pd.Series(y_train).value_counts(sort=False)
pd.Series(y_test).value_counts(sort=False)

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Print a confusion matrix
svc_cm = confusion_matrix(y_train, svc.predict(X_train))
cm_df = pd.DataFrame(svc_cm.T, index=svc.classes_, columns=svc.classes_)
print(cm_df)

svc_cm_test = confusion_matrix(y_test, svc.predict(X_test))
print(pd.DataFrame(svc_cm_test.T, index=svc.classes_, columns=svc.classes_))