# Model evaluation and improvement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

%matplotlib inline

# an example
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# create a syntheticc dataset
X, y = make_blobs(random_state=0)
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression().fit(X_train, y_train)
logreg.score(X_test, y_test)


## cross-validation
# cross validation evaluate generalization performance of a model, which is more stable and thorough than train_test_split.
# the data is split repeatedly and multiple models are trained
# most common cross validation is k-fold cross-validation

mglearn.plots.plot_cross_validation()

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model imp 	LogisticRegression

iris = load_iris()
logreg = LogisticRegression()
# cross_val_score perform three fold by default
scores = cross_val_score(logreg, iris.data, iris.target)
scores









