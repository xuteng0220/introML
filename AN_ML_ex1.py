# import modules
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

# simple functions

# identity matrix
a = np.eye(5)
a
type(a)

# linear regression with one variable
## load and plot data

# load data
data = np.loadtxt('ex1data1.txt', delimiter=',')
# training data
### attention!, ndarray index
x = data[:, [0]]
# target label
y = data[:, [1]]
# number of training data
m = len(x)
m

# plot data
plt.plot(x, y, 'x', c='r');
plt.xlabel('population of city in 10,000s');
plt.ylabel('profit in $10,000s');


# gradient descent

# cost function, $$J(\theta) = \frac{1}{2m}\sum\limits_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$$
# for one variable regression, $$h_\theta(x)=\theta^Tx=\theta_0 + \theta_1x_1$$

## batch gradient decent algorithm
# update theta simultaneously
# first, set an initial theta, learning rate alpha, iteration times t  
# second, update theta with $\theta_i = \theta_i - \frac{\partial}{\partial\theta}J(\theta),\  i = 0, 1$

# add a column of ones
X = np.hstack((np.ones((m, 1)), x))

# initialize theta with (0, 0)
theta = np.zeros((2, 1))
# number of iterations
iteration = 1000
# learning rate
alpha = 0.01


def cost_function(theta, X, y):
# define cost fuction
# theta, regression parameters
# X, training data with 
# y, target label
	h = np.dot((X.dot(theta) - y).T, (X.dot(theta) - y)) / 2 / len(X)
	return h


# print('please enter the iteration numbers:')
# iteration = int(input())
def partial_theta(theta, X, y):
	# partial derivatives of cost fuction
	partial = np.dot(X.T, (X.dot(theta) - y)) / len(X)
	return partial

# iterate gradient decent algorithm
for i in range(iteration):
	theta = theta - alpha * partial_theta(theta, X, y)




# prediction
predict1 = np.array([1, 3.5]).dot(theta)
predict2 = np.array([1, 7]).dot(theta)
predict1
predict2



# plot predictions

# scatter plot
plt.plot(x, y, 'x', c='r');
# predict with theta derived from gradient decent
line = np.linspace(x.min(), x.max(), 100)
predict_line = line * theta[1] + theta[0]
plt.plot(line, predict_line, 'b');
# predict with sklearn linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)
predict_line_lr = lr.predict(line.reshape(-1, 1))
plt.plot(line, predict_line_lr, '--', c='g');


# visiaulize $J(\theta)$

# theta domin
theta0 = np.linspace(-10, 10, 100)
theta1 = np.linspace(-1, 4, 50)

# initialize cost_f with 0's
cost_f = np.zeros((len(theta0), len(theta1)))
# compute cost function
for i in range(len(theta0)):
	for j in range(len(theta1)):
		cost_f[i, j] = cost_function(np.array([[theta0[i]], [theta1[j]]]), X, y)

m, n = np.meshgrid(theta1, theta0)




# 3d plot
from mpl_toolkits.mplot3d import Axes3D
# 定义figure
fig = plt.figure()
# 创建3d图形的两种方式
# 将figure变为3d
ax = Axes3D(fig)

x, y = np.meshgrid(theta0, theta1)
ax.plot_surface(n, m, cost_f)

# contour plot
plt.contourf(n, m, cost_f)
plt.plot(theta[0], theta[1], 'rx')





# multivariable linear regression

# house price data
data1 = np.loadtxt('ex1data2.txt', delimiter=',')
# first two columns are features
x1 = data1[:, :2]
x1.shape
# third column is the target label
y1 = data1[:, [-1]]

# feature normalization
# if not normalize, gradient decent may diverge
# before predicting, the data should be normalized
x1_mean = x1.mean(axis=0)
x1_std = x1.std(axis=0)
X1 = (x1 - x1_mean) / x1_std
# X1.mean(axis=0)
# X1.std(axis=0)

# add a column to x1 with all elements are 1
X1 = np.hstack((np.ones((len(x1), 1)), x1))


# set initial theta, alpha, iteration
theta1 = np.zeros((3, 1))
iteration1 = 1000
alpha1 = 0.01

# iterate gradient decent algorithm
for i in range(iteration1):
	theta1 = theta1 - alpha1 * partial_theta(theta1, X1, y1)




# sklearn linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x1, y1)
lr.coef_
lr.intercept_


# predict with sklearn linear regression
x0 = np.array([2700, 3.1])
lr.predict(x0.reshape(1, -1))
# predict with gradient decent algorithm
x0_pred = np.append(1, (x0 - x1_mean) / x1_std)
x0_pred.dot(theta1)



# selecting learning rate
