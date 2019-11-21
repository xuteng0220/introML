# introduction

## resources
- online course
[course](http://bit.ly/ advanced_machine_learning_scikit-learn)
- code 
[github]( https://github.com/amueller/introduction_to_ml_with_python)

# problems ML can sovle
- supervised ML learn from input and output pairs, especially output labels
	- identify the zip code from handwritten digits(nureuol network?)
	- determine whether a tumor is benigh or malignant based on a medical image(classification)
	- detect fraudulent activity in credit card transactions(calssification)
- unsupervised ML, only the input data is known
	- identify topics in a set of blog posts(unknown topic numbers)
	- segment customers into groups with similar preferences(cluster)
	- detect abnormal access patterns to a website(unknown abnormal pattern numbers?)

## libraries used
import sys
sys.version

import numpy as np
import pandas as pd
import matplotlib
print('matplotlib version: {}'.format(matplotlib.__version__))

import scipy as sp
import IPython
import sklearn
import mglearn
mglearn.__version__

## first example
from sklearn.datasets import load_iris
iris_dataset = load_iris()

### basic acknowledgement of iris data
type(iris_dataset)
iris_dataset.keys()
iris_dataset['DESCR'][:193]
iris_dataset['target_names']
iris_dataset['feature_names']

type(iris_dataset['data'])
iris_dataset['data'].shape
iris_dataset['data'][:5]

# target target_names 映射关系如果自己写怎么写？
type(iris_dataset[target])
iris_dataset['target'].shape
iris_dataset['target']

### split train and test data
from sklearn.model_selection import train_test_split
# parameter random_state shuffles the dataset(pick up train and test set randomly) with a fixed seed for reproduction
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

### first things first, look at the data
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# pd.scatter_matrix?
# mglearn.cm3?
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), maker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

### first ML model, k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
# object knn, 1.encapsulate algorithm to build the model applied on the training data 2. contain algorithm to make predictions on new data set 3.hold the information the algorithm extracted from the training set
knn = KNeighborsClassifier(n_neighbors=1)
# fit method return the knn object
knn.fit(X_train, y_train)


### make predictions
# np.array([5, 2.9, 1, 0.2])?
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print('prediction: {}'.format(prediction))
print('predict target name: {}'.format(iris_dataset['target_names'][prediction]))


### evaluate the model
y_pred = knn.predict(X_test)
y_pred
# score, prediction accuracy, fraction of flowers predicted right
print('test set score: {:.2f}'.format(np.mean(y_pred == y_test)))
print('test set score: {:.2f}'.format(knn.score(X_test, y_test)))


