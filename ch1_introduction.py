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

type(iris_dataset[target])
iris_dataset['target'].shape
iris_dataset['target']


from sklearn.model_selection import train_test_split
# parameter random_state shuffles the dataset(pick up train and test set randomly) with a fixed seed for reproduction
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
X_train.shape
y_train.shape
X_test.shape
y_test.shape



