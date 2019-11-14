# unsupervised learning and preprocessing

import numpy as np
import panda as pd
import matplotlib.pyplot as plt
import mglearn

# unsupervised learning, the learning algorithm is just shown the input data and asked to extract knowledge from this data

## types of unsupervised learning
- dimensionality reduction
- clustering, partition data into distinct groups of similar items

## preprocessing and rescaling
mglearn.plots.plot_scaling()
### applying data transformations
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
X_train.shape
X_test.shape

# basic usage of model in sklearn modules, first import the class, second instantiate it, third fit the data, (forth, transform data for data preprocessing)
# import model
from sklearn.preprocessing import MinMaxScaler
# instantiate
scaler = MinMaxScaler()
# fit
scaler.fit(X_train)

# transform
X_train_scaled = scaler.transform(X_train)
X_train_scaled.shape
X_train.min(axis=0)
X_train.max(axis=0)
X_train_scaled.min(axis=0)
X_train_scaled.max(axis=0)

X_test_scaled = scaler.transform(X_test)
# the test set, after scaling, the min and max are not 0 and 1, because the scaler is based on training set
X_test_scaled.min(axis=0)
X_test_scaled.max(axis=0)


### scaling training and test data the same way
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# test_size?
X_train, X_test = train_test_split(X, random_state=5, test_size=.1) 

fig, axes = plt.subplots(1, 3, figsize=(13, 4))









