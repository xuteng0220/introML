import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

# observe the relationship between features and ouput
train.head(10)
gender_submission.head()
test.head()


train['Pclass'].value_counts()
train['SibSp'].value_counts()
train['Parch'].value_counts()
# train.columns
# select feature columns
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




## tmall gmv
tmall = {'year': [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018], 'gmv': [0.5, 9.36, 52, 191, 350, 571, 912, 1207, 1682, 2135]} #2019,2684
tmall11 = pd.DataFrame(tmall)
X = tmall11.iloc[:, 0].values.reshape(-1, 1)
y = tmall11.iloc[:, 1].values.reshape(-1, 1)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_poly, y)
lr.coef_
lr.intercept_

test_poly = poly.transform(2019)
lr.predict(test_poly)
