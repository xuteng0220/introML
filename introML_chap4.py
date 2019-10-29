import numpy as np
import pandas as pd
import mglearn
# representing data and engineering features


## categorical variables
# load data, first find it in the original data file
import pandas as pd
data = pd.read_csv('adult.data', header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occuption', 'relationship',''])
## binning
## interactions and polynomials
## univariate nonlinear transformation



## feature selection
# reduce the number of features to simplify models that generalize better


### univariate statistics
# supervised method, need the target label, split the data into train and test
# compute a statistically significant relationship between each feature and the target. use a test to compute the relationship, use a method for discarding features with high p-value
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

# get deterministic random numbers
rng = np.random.RandomState(42)

# add noise feature to the original data
noise = rng.normal(size=(len(cancer.data), 50))
X_w_noise = np.hstack([cancer.data, noise])

# test_size?
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

# with f_classif (default relationship test), select 50% features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# transform train data
X_train_selected = select.transform(X_train)

X_train.shape
X_train_selected.shape

# get_support get the features selected
mask = select.get_support()
mask
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('sample index')
plt.show()


from sklearn.linear_model import LogisticRegression
# transform test datasets
X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print('score with all features: {:.3f}'.format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print('score with selected features:{:.3f}'.format(lr.score(X_test_selected, y_test)))

### model-based selection
# supervised method, need the target label, split the data into train and test
# selevtion model provide measurements for all fearture at once, rank the measurements, discard features
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
# SelectFromModel is a transformer with a threshold select the features having an importance measure larger than the threshold, RandomForestClassifier is a supervised model providing fearture importance
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
X_train.shape
X_train_l1.shape

mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('sample index')

X_test_l1 = select.transform(X_test)
# score on test data after model-based feature selection
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
score
### iterative selection
# supervised method, need the target label, split the data into train and test
# a series models are built with varying numbers of features. two methods, 1) add 1 feature at o time until some stopping criterion is reached. 2) eliminate 1 feature at a time with all features until some stopping criterion is reached

# RFE(recursive feature elimination), start with all features, using a specific model, discard one with least importance, and repeat until prespecified number of features are left
from sklearn.feature_selection import RFE
select = FRE(RandomForestClassifier(m n_estimators=100, random_state=42), n_feature_to_select=40)
select.fit(X_train, y_train)
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('sample index')

X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
# score on test data after iterative feature selection
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
score

# use the model(use only the features seleted) used inside the RFE to predict on test data
select.score(X_test, y_test)






## utilizing expert konwleadge
# prior knowledge of the data which cannot be captured by models from initial representation of data

citibike = mglearn.datasets.load_citibike()

citibike.head()
plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
# ha ?
plt.xticks(xticks, xticks.strftime('% a %m-%d'), rotation=90, ha='left')
plt.plot(citibike, linewidth=1)
plt.xlabel('Date')
plt.ylabel('Rentals')

y = citibike.values
x = citibike.index.strftime
