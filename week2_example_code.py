# -*- coding: utf-8 -*-
"""
@author: written by Mei-Chen
2021 March

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as skl_lm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

yourpath = 'C:/Users/ntpu_metrics/Documents/Python Scripts/Datasets_ISLR/'

#Example: the Advertising data
advertising = pd.read_csv(str(yourpath)+'Advertising.csv', usecols=[1,2,3,4])  
advertising.info()
advertising.head(3)

#advertising: Ordinary Least Squares
#y ~ a + a:b + np.log(x), a:d = interaction
est = smf.ols('sales ~ TV', advertising).fit()  
est.summary()

est = smf.ols('sales ~ radio', advertising).fit()
est.summary().tables[1]  #call the second table with .tables[1]

est = smf.ols('sales ~ newspaper', advertising).fit()
est.summary().tables[1]

est = smf.ols('sales ~ TV + radio + newspaper', advertising).fit()
est.summary()
advertising.corr()

est = smf.ols('sales ~ TV + radio + TV*radio', advertising).fit()
est.summary()

#advertising: sklearn
regr = skl_lm.LinearRegression()
features_cols =['TV','radio','newspaper']
X = advertising[features_cols]
y = advertising['sales']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=80)

regr.fit(x_train,y_train)
print([regr.intercept_, regr.coef_])

y_predict = regr.predict(x_test)
print(y_predict)

plt.figure()
plt.plot(range(len(y_predict)),y_predict,'b',label='predict')
plt.plot(range(len(y_predict)),y_test,'r',label='test')
plt.legend(loc="upper right")
plt.xlabel("the number of sales")
plt.ylabel("value of sales")
plt.show()

plt.scatter(x_test['TV'], y_test,  color='black')
plt.scatter(x_test['TV'], y_predict,  color='red')
plt.xticks(())
plt.yticks(())
plt.show()

#Example: the Auto data
auto = pd.read_csv(str(yourpath)+'Auto.csv', na_values='?').dropna()
print(auto.info())
print(auto.head(5))
print(auto.describe())

est = smf.ols('mpg ~ horsepower + np.power(horsepower, 2)', auto).fit()
est.summary()

plt.scatter(auto.horsepower, auto.mpg, facecolors='None', edgecolors='k', alpha=.5) 
sns.regplot(auto.horsepower, auto.mpg, ci=100, label='Linear', scatter=False, color='orange')
#If ``order`` is greater than 1, use ``numpy.polyfit`` to estimate polynomial regression
sns.regplot(auto.horsepower, auto.mpg, ci=100, label='Degree 2', order=2, scatter=False, color='lightblue')
sns.regplot(auto.horsepower, auto.mpg, ci=100, label='Degree 5', order=5, scatter=False, color='g')
plt.legend()  #locating labels, loc='upper right'
plt.ylim(5,55)
plt.xlim(40,240)
plt.show()

#The Validation Set Approach
train_auto2 = auto.sample(196, random_state = 2)
test_auto2 = auto[~auto.isin(train_auto2)].dropna(how = 'all')

X_train = train_auto2['horsepower'].values.reshape(-1,1)
y_train = train_auto2['mpg']
X_test = test_auto2['horsepower'].values.reshape(-1,1)
y_test = test_auto2['mpg']

lm = skl_lm.LinearRegression()

# Linear
model = lm.fit(X_train, y_train)
print(mean_squared_error(y_test, model.predict(X_test)))

# Quadratic
poly = PolynomialFeatures(degree=2)
X_train2 = poly.fit_transform(X_train)
X_test2 = poly.fit_transform(X_test)

model = lm.fit(X_train2, y_train)
print(mean_squared_error(y_test, model.predict(X_test2)))

# Cubic
poly = PolynomialFeatures(degree=3)
X_train3 = poly.fit_transform(X_train)
X_test3 = poly.fit_transform(X_test)

model = lm.fit(X_train3, y_train)
print(mean_squared_error(y_test, model.predict(X_test3)))

#LOOCV
lm = skl_lm.LinearRegression()
model = lm.fit(X_train, y_train)

loo = LeaveOneOut()
X = auto['horsepower'].values.reshape(-1,1)
y = auto['mpg'].values.reshape(-1,1)
loo.get_n_splits(X)

cv_origin = KFold(n_splits=392, random_state=None, shuffle=False)  #total 392 observations
#shuffle=False means the random_state is integer and result will be the same
scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv_origin, n_jobs=1)
#estimator = model
print("Folds: " + str(len(scores)) + ", MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))

#LOOCV: polynomial regression
#fit(): Method calculates the parameters μ and σ and saves them as internal objects
#transform(): Method using these calculated parameters apply the transformation to a particular dataset
#fit_transform(): joins the fit() and transform() method for transformation of dataset.
for i in range(1,6):
    poly = PolynomialFeatures(degree=i)
    X_current = poly.fit_transform(X)
    model = lm.fit(X_current, y)
    scores = cross_val_score(model, X_current, y, scoring="neg_mean_squared_error", cv=cv_origin,
 n_jobs=1)
    
    print("Degree-"+str(i)+" polynomial MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))
    
#k-fold
kf = KFold(n_splits=10, random_state=1, shuffle=False) #k = 10:a common choice for k

for i in range(1,11):
    poly = PolynomialFeatures(degree=i)
    X_current = poly.fit_transform(X)
    model = lm.fit(X_current, y)
    scores = cross_val_score(model, X_current, y, scoring="neg_mean_squared_error", cv=kf,
 n_jobs=1)
    
    print("Degree-"+str(i)+" polynomial MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))
    
