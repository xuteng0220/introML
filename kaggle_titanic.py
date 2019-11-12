import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

train.head()
gender_submission.head()
test.head()

# train.columns
# select 'PassengerId', 'Pclass', 'SibSp', 'Parch', 'Ticket', 'Fare'
X_train = train.iloc[:, [0, 2, 6, 7, 9]].values
y_train = train.iloc[:, 1]

X_test = test.iloc[:, [0, 1, 5, 6, 8]].dropna().values

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.predict(X_test)



# there is nan value in both the train and test set
# how to add categorical variables
