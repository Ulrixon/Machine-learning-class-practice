# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:05:59 2021

@author: ntpu_metrics
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as skl_lm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score, roc_curve, auc, log_loss
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.feature_selection import f_regression
from sklearn.metrics import confusion_matrix
from scipy import stats
import scikitplot as skplt  #a visualisation tool from Scikit-Learn + Matplotlib for ML
import statsmodels.api as sm
import statsmodels.formula.api as smf
from ipywidgets import widgets  #interact with notebook

%matplotlib inline
plt.style.use('seaborn-white')

###Define confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

 

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#Example: the stock market data (Logistic Regression)
###Load Datasets
df_stock = pd.read_csv('C:/Users/ntpu_metrics/Documents/Python Scripts/Datasets_ISLR/Smarket.csv', index_col=0,parse_dates=True)
# convert direction to binary. Up is 1, Down is 0
df_stock.replace({'Up': 1, 'Down': 0}, inplace=True)
df_stock.head(3)
df_stock.describe()
df_stock.corr()

# very small correlations (today and direction are obiously correlated)
corr = df_stock.corr().values
print(corr)
np.max(np.abs(np.triu(corr, k=1)), axis=1)
#triu for returning Upper triangle of an array

# volume increases with year
plot = sns.boxplot(df_stock.index, df_stock['Volume'],)
plot.set_xticklabels([str(date.year) for date in df_stock.index.unique()])

# volumne by year and direction
ax = sns.boxplot(df_stock.index, df_stock['Volume'], hue=df_stock['Direction'])
ax.set_xticklabels([str(date.year) for date in df_stock.index.unique()])  #show year only
handles, _ = ax.get_legend_handles_labels()  #control labels via handle
ax.legend(handles, ["Down", "Up"])

###split X/y train/test
#df.column.difference() create a new dataframe from an existing dataframe with exclusion of some columns
X = df_stock[df_stock.columns.difference(['Today', 'Direction'])]
y = df_stock['Direction']
X_train = X[:'2004'] #2001-2004
y_train = y[:'2004']
X_test = X['2005':] #2005
y_test = y['2005':]

###Logistic regression with test/train split
logistic_test = skl_lm.LogisticRegression(C=1e10)
logistic_test.fit(X_train, y_train)
#coef_&intercept_ 
print("coef: ", logistic_test.coef_)
print("intercept: ", logistic_test.intercept_)
#p-value(f_regression from sklearn.feature_selection)
print("p-value: ", f_regression(X_train, y_train)[1])
#accuracy
logistic_pre = logistic_test.predict(X_train)
accuracy = logistic_test.score(X_train, y_train)
print("accuracy: ",accuracy)

#confusion matrix
cnf2=confusion_matrix(y_test,logistic_test.predict(X_test))
print('confusion_matrix: ',cnf2)

target_name=['Down','Up']
plot_confusion_matrix(cnf2,classes=target_name,title='confusion_matrix')
plt.show()













