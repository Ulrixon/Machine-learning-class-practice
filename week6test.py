# %%
import numpy
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from numpy import linalg
import statistics 

# %%
data1 =  pd.read_excel(r'/Users/ryan/Downloads/example.xls', sheet_name=0, header=0)
data1.index = pd.to_datetime(data1.iloc[:,0])
data1 = data1.drop('date',axis=1)
idx = np.where(data1.iloc[:,0]<17,1,0)
df_yes = data1[idx == 1] # recession
df_no = data1[idx == 0]
# %%
uhat_yes=  numpy.mean(df_yes.iloc[:,1:3],axis=0)
covhat_yes= numpy.cov(df_yes.iloc[:,1:3].T)
uhat_no=  numpy.mean(df_no.iloc[:,1:3],axis=0)
covhat_no= numpy.cov(df_no.iloc[:,1:3].T)
py_no=numpy.mean(idx)
py_yes=1-numpy.mean(idx)
# %%
deta_ys=(uhat_yes.T.dot(np.linalg.inv(covhat_yes)).dot(data1.iloc[:,1:3].T)
            -uhat_yes.T.dot(np.linalg.inv(covhat_yes)).dot(uhat_yes))+math.log(py_yes)
deta__no=(uhat_no.T.dot(np.linalg.inv(covhat_no)).dot(data1.iloc[:,1:3].T)
            -uhat_no.T.dot(np.linalg.inv(covhat_no)).dot(uhat_no))+math.log(py_no)
deta_c=  numpy.vstack((deta_ys,deta__no)).T
# %%
