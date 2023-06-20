#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:12:33 2019

@author: kian
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

# %%
# example
X = pd.read_excel(r"/Users/ryan/Documents/vscode(py)/ndc.xls", sheet_name=0, header=0,)
X.index = pd.to_datetime(X.iloc[:, 0])
X = X.drop("date", axis=1)
y = pd.read_excel(
    r"/Users/ryan/Documents/vscode(py)/ndc_light.xls", sheet_name=0, header=0,
)
y.index = pd.to_datetime(y.iloc[:, 0])
y = y.drop("date", axis=1)

data = pd.concat([X, y], axis=1)
data_train, data_test = train_test_split(data, train_size=100)

X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
X_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]


# first tree model
# %%
clf0 = tree.DecisionTreeRegressor(min_samples_split=5)
model0 = clf0.fit(X=X_train, y=y_train)
y0_hat = clf0.predict(X_test)
mse0 = np.mean((y0_hat - y_test) ** 2)

dot_data = tree.export_graphviz(model0, feature_names=X.columns)
graph = graphviz.Source(dot_data)
graph


# tree model with different depths
parameters = {"max_depth": range(1, 10)}
clf = GridSearchCV(
    tree.DecisionTreeRegressor(),
    parameters,
    n_jobs=4,
    cv=5,
    scoring="neg_mean_squared_error",
)
GCV_model = clf.fit(X=X_train, y=y_train)
print("The best depth is", GCV_model.best_params_["max_depth"])
plt.plot(GCV_model.cv_results_["mean_test_score"] * (-1))

mse = np.zeros((9, 2))
for ii in range(1, 10):
    clf0 = tree.DecisionTreeRegressor(max_depth=ii)
    model0 = clf0.fit(X=X_train, y=y_train)
    y0_hat = model0.predict(X_test)
    mse[ii - 1, 0] = np.mean((y0_hat - y_test) ** 2)
    y0_hat = model0.predict(X_train)
    mse[ii - 1, 1] = np.mean((y0_hat - y_train) ** 2)


fig0, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=200)
ax.plot(GCV_model.cv_results_["mean_test_score"] * (-1), label="CV")
ax.plot(mse[:, 0], label="Testing error")
ax.plot(mse[:, 1], label="Training error")
ax.legend(loc="upper left")
plt.show()


tree_model = clf.best_estimator_
dot_data = tree.export_graphviz(tree_model, feature_names=X.columns)
graph = graphviz.Source(dot_data)
graph


# Bagging
clf_bag = BaggingRegressor(
    tree.DecisionTreeRegressor(max_depth=3), n_estimators=200, random_state=10
)
clf_estimator = clf_bag.fit(X=X_train, y=y_train)
y_bag_hat = clf_bag.predict(X_test)
mse_bag = np.mean((y_bag_hat - y_test) ** 2)

# different tree in bagging
dot_data = tree.export_graphviz(clf_estimator.estimators_[3], feature_names=X.columns)
graph = graphviz.Source(dot_data)
graph


clf_rf = RandomForestRegressor(min_samples_split=5, random_state=10)
clf_rf.fit(X=X_train, y=y_train)
clf_rf.predict(X_test)
y_rf_hat = clf_rf.predict(X_test)
mse_rf = np.mean((y_rf_hat - y_test) ** 2)
# %%
# Random forests
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
# plt.rcParams['font.sans-serif']=['SimHei'] #Show Chinese label
plt.rcParams["axes.unicode_minus"] = False

features = X.columns
importances = clf_rf.feature_importances_
indices = np.argsort(importances)
plt.figure()
plt.title("Feature importances (random forest)")
plt.barh(range(len(indices)), importances[indices], color="b", align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.show()


# %%
