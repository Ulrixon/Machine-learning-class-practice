# %% import
from copy import deepcopy
from itertools import chain, combinations
from re import I
from statistics import mean
from cmath import log
from unicodedata import numeric

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from numpy import argmin, linalg
from numpy.linalg import inv
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# %% import data
data1 = pd.read_excel(r"/Users/ryan/Downloads/data4.xlsx", sheet_name=0, header=0)
xdata1 = data1.iloc[range(300), range(2, 12)]
# %% c10 取ii list
xcombination = []
for ii in range(1, 11):
    xcombination.append(list(combinations(range(10), ii)))
# %%
for i in range(10):

    plt.hist(xdata1.iloc[:, i])
    plt.title(i + 1)
    plt.show()
plt.hist(xdata1.iloc[:, 9].apply(lambda x: log(x)))
#%% question 1 ans:
# 因為x變數在某些x上不是常態,而且qda,lda跑出的
# 最好模型也包含那些非常態變數，所以用logit模型最適合
# 不過我還是都跑跑看，順帶一提變異數也明顯不同且cv值最低的不是logit而是qda。
# %% LDA+LOOCV
ldacvindex = deepcopy(xcombination)
for j in range(10):
    for k in range(int(len(xcombination[j]))):

        X = xdata1.iloc[:, list(map(int, xcombination[j][k]))]
        y = data1.loc[::, "y"]
        CV = 0
        # loocv
        for l in range(300):
            lda = LinearDiscriminantAnalysis()
            Xexclude = X.iloc[chain(range(l), range(l + 1, 300)), ::]
            yexclude = y[chain(range(l), range(l + 1, 300))]
            yout = X.iloc[l, ::]
            ypred = lda.fit(Xexclude.values, yexclude.values).predict(
                np.reshape(X.loc[l].to_numpy(), (1, j + 1))
            )
            CV = CV + (y[l] - ypred[0]) ** 2

        ldacvindex[j][k] = CV / 300

# %% Find minimum cv location (0, 4, 6, 7, 9) 0.18333333333333332
mini = list()
minilocate = list()
for i in range(10):
    mini.append(min(ldacvindex[i]))
    minilocate.append(np.argmin(ldacvindex[i]))
ldamincombin = xcombination[np.argmin(mini)][minilocate[np.argmin(mini)]]
# %% QDA +LOOCV
qdacvindex = deepcopy(xcombination)
for j in range(10):
    for k in range(int(len(xcombination[j]))):

        X = xdata1.iloc[:, list(map(int, xcombination[j][k]))]
        y = data1.loc[::, "y"]
        CV = 0
        for l in range(300):
            qda = QuadraticDiscriminantAnalysis()
            Xexclude = X.iloc[chain(range(l), range(l + 1, 300)), ::]
            yexclude = y[chain(range(l), range(l + 1, 300))]
            yout = X.iloc[l, ::]
            ypred = qda.fit(Xexclude.values, yexclude.values).predict(
                np.reshape(X.loc[l].to_numpy(), (1, j + 1))
            )
            CV = CV + (y[l] - ypred[0]) ** 2

        qdacvindex[j][k] = CV / 300
# %% . (1, 3, 6, 7, 9) 0.17666666666666667
mini = list()
minilocate = list()
for i in range(10):
    mini.append(min(qdacvindex[i]))
    minilocate.append(np.argmin(qdacvindex[i]))
qdamincombin = xcombination[np.argmin(mini)][minilocate[np.argmin(mini)]]
# %% logistic reg. +loocv
logcvindex = deepcopy(xcombination)
for j in range(10):
    for k in range(int(len(xcombination[j]))):
        loocv = LeaveOneOut()

        X = xdata1.iloc[:, list(map(int, xcombination[j][k]))]
        y = data1.loc[::, "y"]
        CV = cross_val_score(
            LogisticRegression(solver="lbfgs", n_jobs=-1),
            X,
            y,
            cv=loocv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        #        CV = 0
        #       for l in range(300):

        #            Xexclude = X.iloc[chain(range(l), range(l + 1, 300)), ::]
        #            yexclude = y[chain(range(l), range(l + 1, 300))]
        #            yout = X.iloc[l, ::]
        #            ypred = (
        #                LogisticRegression()
        #                .fit(Xexclude.values, yexclude.values)
        #                .predict(np.reshape(X.loc[l].to_numpy(), (1, j + 1)))
        #            )
        #            CV = CV + (y[l] - ypred[0]) ** 2

        logcvindex[j][k] = CV
# %% (0, 3, 6, 7, 9) 0.18333333333333332
mini = list()
minilocate = list()
for i in range(10):
    mini.append(min(logcvindex[i]))
    minilocate.append(np.argmin(logcvindex[i]))
logmincombin = xcombination[np.argmin(mini)][minilocate[np.argmin(mini)]]
np.min(mini)
# %%
def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(1, 3, fig_index)
    if fig_index == 1:
        plt.title("Linear Discriminant Analysis", fontsize=20)
    elif fig_index == 2:
        plt.title("Quadratic Discriminant Analysis", fontsize=20)
    elif fig_index == 3:
        plt.title("Logistic regression Analysis", fontsize=20)
    tp = y == y_pred  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker=".", color="red")
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker="x", s=20, color="#990000")  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker=".", color="blue")
    plt.scatter(
        X1_fp[:, 0], X1_fp[:, 1], marker="x", s=20, color="#000099"
    )  # dark blue

    return splot


def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(
        mean,
        2 * v[0] ** 0.5,
        2 * v[1] ** 0.5,
        180 + angle,
        facecolor=color,
        edgecolor="black",
        linewidth=2,
    )
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.2)
    splot.add_artist(ell)


def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, "red")
    plot_ellipse(splot, lda.means_[1], lda.covariance_, "blue")


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariance_[0], "red")
    plot_ellipse(splot, qda.means_[1], qda.covariance_[1], "blue")


# %%
X = xdata1.iloc[:, list(ldamincombin)].values
Xqda = xdata1.iloc[:, list(qdamincombin)].values
Xlogit = xdata1.iloc[:, list(logmincombin)].values
# for i, (X, y) in enumerate([(X,y)]):
# Linear Discriminant Analysis
logregress = LogisticRegression(n_jobs=-1)
y_pred = logregress.fit(Xlogit, y).predict(Xlogit)
splot = plot_data(lda, Xlogit, y, y_pred, fig_index=3)  # 2 * i + 1)

plt.xlabel("x1", fontsize=20)
plt.ylabel("x4", fontsize=20)
plt.axis("tight")


plt.tight_layout()
plt.show()
# %% confusionmatrix plot
def confusionmatrixplot(qdaconfusionmatrix):
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in qdaconfusionmatrix.flatten()]
    group_percentages = [
        "{0:.2%}".format(value)
        for value in qdaconfusionmatrix.flatten() / np.sum(qdaconfusionmatrix)
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    ax = sns.heatmap(qdaconfusionmatrix, annot=labels, fmt="", cmap="Blues")
    ax.set_title("Seaborn Confusion Matrix with labels\n\n")
    ax.set_xlabel("\nPredicted Values")
    ax.set_ylabel("Actual Values ")
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(["False", "True"])
    ax.yaxis.set_ticklabels(["False", "True"])
    ## Display the visualization of the Confusion Matrix.
    plt.show()


# %% lda confusion
lday_pred = lda.fit(X, y).predict(X)
ldaconfusionmatrix = confusion_matrix(y, lday_pred)

print("confusion_matrix: ", ldaconfusionmatrix)

# %% qda confusion
qday_pred = qda.fit(Xqda, y).predict(Xqda)
qdaconfusionmatrix = confusion_matrix(y, qday_pred)
print("confusion_matrix: ", qdaconfusionmatrix)
confusionmatrixplot(qdaconfusionmatrix)
# %% log confusion
log_pred = logregress.fit(Xlogit, y).predict(Xlogit)
logconfusionmatrix = confusion_matrix(y, log_pred)
print("confusion_matrix: ", qdaconfusionmatrix)

confusionmatrixplot(logconfusionmatrix)
# %% caculate FDR...
confusion = logconfusionmatrix
TN = confusion[0][0]
FN = confusion[1][0]
TP = confusion[1][1]
FP = confusion[0][1]
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
# Specificity or true negative rate
TNR = TN / (TN + FP)
# Precision or positive predictive value
PPV = TP / (TP + FP)
# Negative predictive value
NPV = TN / (TN + FN)
# Fall out or false positive rate
FPR = FP / (FP + TN)
# False negative rate
FNR = FN / (TP + FN)
# False discovery rate
FDR = FP / (TP + FP)
print(FDR)
# Overall accuracy
ACC = (TP + TN) / (TP + FP + FN + TN)
# %%

