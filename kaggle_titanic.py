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
X_train = train.loc[:, ['PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch']]
y_train = train.loc[:, 'Survived']

X_test = test.loc[:, ['PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch']]

# get dummy variables
X_train_dy = pd.get_dummies(X_train).values
X_test_dy = pd.get_dummies(X_test).values



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.1, penalty="l1")
logreg.fit(X_train_dy, y_train)
logreg.score(X_train_dy, y_train)
logreg.predict(X_test_dy)



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


# plot the gmv
