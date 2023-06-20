# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:34:54 2021

@author: ntpu_metrics
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import graphviz
#import pydotplus
%matplotlib inline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.pipeline import make_pipeline

path = r'/home/kian/Dropbox/NTPU/Course/2021Spring/MachineLearning/Week9/datasets'


#8.3.2 Regression tree
boston = pd.read_csv(str(yourpath)+'Boston.csv').drop('Unnamed: 0', axis=1)
X = boston.drop('medv', axis=1)
y = boston.medv
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=0)

# Choosing max depth 2
regr = DecisionTreeRegressor(max_depth=2)
regr.fit(X_train, y_train)

#plot tree
dot_data = export_graphviz(regr, out_file=None, feature_names=X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf(str(yourpath)+'Boston.pdf')

pred = regr.predict(X_test)

#find the best depth, using grid search
#with pipeline:  you need to name steps but name doesn't change if you change estimator/transformer used in a step.
pipe_tree = make_pipeline(DecisionTreeRegressor(random_state=1)) #make pipeline: generates names for steps automatically
depths = np.arange(1, 21)
num_leafs = [1, 5, 10, 20, 50, 100]
param_grid = [{'decisiontreeregressor__max_depth':depths, #using __ to build parameters
              'decisiontreeregressor__min_samples_leaf':num_leafs}]
gs = GridSearchCV(estimator=pipe_tree, param_grid=param_grid, cv=10)
gs = gs.fit(X_train, y_train)
tree_model = gs.best_estimator_
print (gs.best_score_, gs.best_params_) 

#find the best depth, using CV
depth = []
for i in range(2,20):
    regr = DecisionTreeRegressor(max_depth=i)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=regr, X=X_train, y=y_train, cv=10)
    depth.append((i,scores.mean()))
print(depth)


#predict test data
plt.scatter(pred, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

mean_squared_error(y_test, pred)



#8.3.2 Bagging and Random Forests
# Bagging: using all features
#bagging is simply a special case of a random forest with $m = p$. 
#Therefore, the ${\tt RandomForestRegressor()}$ function can be used to perform both random forests and bagging.
regr1 = RandomForestRegressor(max_features=13, random_state=1)
regr1.fit(X_train, y_train)

pred = regr1.predict(X_test)
plt.scatter(pred, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')
mean_squared_error(y_test, pred)

# Random forests: using 6 features, using a smaller value of the ${\tt max\_features}$ argument
regr2 = RandomForestRegressor(max_features=6, random_state=1)
regr2.fit(X_train, y_train)

pred = regr2.predict(X_test)
mean_squared_error(y_test, pred)

Importance = pd.DataFrame({'Importance':regr2.feature_importances_*100}, index=X.columns)
Importance.sort_values(by='Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None

