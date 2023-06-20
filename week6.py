#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:56:19 2019

@author: kian
"""
# %%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import linalg


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

%matplotlib inline
plt.style.use('seaborn-white')

# %%
# example
data1 =  pd.read_excel(r'/Users/ryan/Downloads/example.xls', sheet_name=0, header=0)
data1.index = pd.to_datetime(data1.iloc[:,0])
data1 = data1.drop('date',axis=1)
idx = np.where(data1.iloc[:,0]<17,1,0)
df_yes = data1[idx == 1] # recession
df_no = data1[idx == 0]
data1['index'].plot()

# %%

fig = plt.figure(figsize=(8,8),dpi=100)
ax1 = plt.subplot(1,1,1)
ax1.scatter(df_yes.m1b, df_yes.m2, s=40, c='orange', marker='+',linewidths=1)
ax1.scatter(df_no.m1b, df_no.m2, s=40, marker='o', linewidths=1,edgecolors='blue', facecolors='white', alpha=.6)
ax1.set_xlabel('m1b',fontsize=16)
ax1.set_ylabel('m2',fontsize=16)
ax1.legend(['<17', '>17'],fontsize=16)


X = data1[['m1b', 'm2']]
y = idx

# %%
lda = LinearDiscriminantAnalysis()
y_pred = lda.fit(X, y).predict(X)

df_ = pd.DataFrame({'True recession status': y,
                    'Predicted recession status': y_pred})
df_.replace(to_replace={0:'No', 1:'recession'}, inplace=True)

df_.groupby(['Predicted recession status','True recession status']).size().unstack('True recession status')
# %%
qda = QuadraticDiscriminantAnalysis()
y_pred2 = qda.fit(X, y).predict(X)
df2_ = pd.DataFrame({'True recession status': y,
                    'Predicted recession status': y_pred2})
df2_.replace(to_replace={0:'No', 1:'recession'}, inplace=True)

df2_.groupby(['Predicted recession status','True recession status']).size().unstack('True recession status')

# %%
# #############################################################################
# Generate datasets
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


def dataset_cov():
    '''Generate 2 Gaussians samples with different covariance matrices'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

# %%
# #############################################################################
# Plot functions
def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(1, 2, fig_index)
    if fig_index == 1:
        plt.title('Linear Discriminant Analysis',fontsize=20)
    elif fig_index == 2:
        plt.title('Quadratic Discriminant Analysis',fontsize=20)


    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')

    return splot


def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, facecolor=color,
                              edgecolor='black', linewidth=2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.2)
    splot.add_artist(ell)


def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariance_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariance_[1], 'blue')


# %%
from matplotlib import colors
# #############################################################################
# Colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

X=X.values

plt.figure(figsize=(15, 8), facecolor='white')
for i, (X, y) in enumerate([(X,y)]):
    # Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
    plot_lda_cov(lda, splot)
    plt.xlabel('m1b',fontsize=20)
    plt.ylabel('m2',fontsize=20)
    plt.axis('tight')

    # Quadratic Discriminant Analysis
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    y_pred = qda.fit(X, y).predict(X)
    splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
    plot_qda_cov(qda, splot)
    plt.xlabel('m1b',fontsize=20)
    plt.ylabel('m2',fontsize=20)
    plt.axis('tight')
plt.suptitle('Linear Discriminant Analysis vs Quadratic Discriminant Analysis',
             y=1.02, fontsize=20)
plt.tight_layout()
plt.show()







# %%
