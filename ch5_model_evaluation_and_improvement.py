# Model evaluation and improvement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

# an example
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_blobs(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression().fit(X_train, y_train)
logreg.score(X_test, y_test)


## cross-validation
