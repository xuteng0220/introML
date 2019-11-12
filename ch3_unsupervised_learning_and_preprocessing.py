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
## applying data transformations
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
X_train.shape
X_test.shape

# basic usage of model in sklearn modules, first import the class, second instantiate it, then fit the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)


train_test_split