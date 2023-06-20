# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
np.random.seed(20)
N = 500
k = 6
beta0 = np.array([[0.9,0.5,0.2,-0.2,-0.7,0.3]]).T
X = np.random.randn(N,k)
e = 2*np.random.randn(N,1)
y = X@beta0+e
bhat = inv(X.T@X)@X.T@y
print(bhat)

bhat0 = np.zeros([k,1])
cost=(y-X@bhat0[:,0:1]).T@(y-X@bhat0[:,0:1])/N
eta = 0.001
criterion = 0.000000001
max_iters = 20000
ii = 0
dd = 999
while dd > criterion and ii < max_iters:
    g = -X.T@(y-X@bhat0[:,ii:ii+1])/N
    d=(g>0)*-1+(g<0)*1
    #bhat0 = np.append(bhat0, bhat0[:,ii:ii+1]-eta*g, axis=1)
    bhat0 = np.append(bhat0, bhat0[:,ii:ii+1]-eta*g, axis=1)
    e2 = (y-X@bhat0[:,ii+1:ii+2]).T@(y-X@bhat0[:,ii+1:ii+2])/N
    cost = np.append(cost,e2, axis=1)
    dd = abs((cost[0,ii+1]-cost[0,ii])/cost[0,ii])
    ii += 1
fig1 = plt.figure(figsize=(5,4), dpi=200)
ax = fig1.add_subplot(1,1,1)
ax.plot(bhat0[0,:])
ax.axhline(y=0.9, linewidth=1, color = 'r')
ax.tick_params(labelsize=16)
ax.set_xlabel('iterations',fontsize=16)
ax.set_ylabel(r"$\beta_1$",fontsize=16)
plt.show()

fig2 = plt.figure(figsize=(5,4), dpi=200)
ax = fig2.add_subplot(1,1,1)
ax.plot(bhat0[4,:])
ax.axhline(y=-0.7, linewidth=1, color = 'r')
ax.tick_params(labelsize=16)
ax.set_xlabel('iterations',fontsize=16)
ax.set_ylabel(r"$\beta_5$",fontsize=16)
plt.show()
print(bhat0[:,ii])
# %%
