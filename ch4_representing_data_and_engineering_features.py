# representing data and engineering features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

%matplotlib inline

# 使用分号; 结束这条线。这会在生成绘图时抑制不需要的输出
# plt.plot(np.arange(10), np.random.randn(10), 'b');

# represent data the best way for a paticular application is called feature engineering


## categorical variables


### one hot encoding（独热编码）(one-out-of_N encoding, dummy variables)
# dummy variables is to replace a categorical variable with one or more new features only have values 0 and 1

# load data, https://github.com/amueller/introduction_to_ml_with_python
# chrome open the txt file, and save as
data = pd.read_csv('adult.csv', header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship','race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

# dropna in case the data not fit for ML algorithm
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']].dropna()
data.head()


data.gender.value_counts()
data.columns

# pandas' function get_dummies transform columns with string or categorical values to dummies (not for integer values)
data_dummies = pd.get_dummies(data)
data_dummies.columns
data_dummies.head()

# pandas column indexing will include the end of the range, here is occupation_ Transport-moving, while numpy will exclude it
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
features.columns

# transform pandas' DataFrame to Numpy array for some certain ML models

X = features.values
y = data_dummies['income_ >50K'].values
print('X.shape: {}, y.shape: {}'.format(X.shape, y.shape))

### classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.score(X_test, y_test)

### numbers can be encoded as categoricals
# OneHotEncoder in sklearn can convert numeric columns to dummy variables

demo_df = pd.DataFrame({'integer feature': [0, 1, 2, 1], 'categorical feature': ['socks', 'fox', 'socks', 'box']})
demo_df

# get_dummies only works on categorical strings, leaving integers unchanged
pd.get_dummies(demo_df)
# transform integer to strings
demo_df['integer feature'] = demo_df['integer feature'].astype(str)
# list columns to be encoded
pd.get_dummies(demo_df, columns=['integer feature', 'categorical feature'])
# the same as 
# pd.get_dummies(demo_df)

## binning
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
X, y = mglearn.datasets.make_wave(n_samples=100)
X.shape
y[:5]
# X.min()
# X.max()
# (3, 3) matches the max and min of X
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
line.shape

# decision tree and linear regression without binning
# linear regression
reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label='decision tree')
# decision tree regression
reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label='linear regression')
# X shape (100, 1), plt.plot is the same as plt.scatter
plt.plot(X, y, 'o', c='k')
plt.ylabel('regression output')
plt.xlabel('input feature')
plt.legend(loc='best')


bins = np.linspace(-3, 3, 11)
bins
print('bins: {}'.format(bins))

# np.digitize find which bin a certain number falls into
which_bin = np.digitize(X, bins=bins)
print('\nData points:\n', X[:5])
print('\nBin membership for data points:\n', which_bin[:5])

# OneHotEncoder from the preprocessing module, it only works on integer categorical variables
from sklearn.preprocessing import OneHotEncoder
# return sparse matrix if true else an array
encoder = OneHotEncoder(sparse=False)

# OneHotEncoder transform bin to dummy variables
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
X_binned[:5]
X_binned.shape
## X_binned shape (100, 10)
## there are 10 bins 
# np.unique(which_bin)

# np.digitize which bin each element in line falls in, encode it as dummy variables
line_binned = encoder.transform(np.digitize(line, bins=bins))
# linear regression
reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')
# decision tree regression
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), '--', label='decision tree binned')
plt.plot(X, y, 'o', c='k')
# plt.vlines, vertical lines
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc='best')
plt.ylabel('regression output')
plt.xlabel('input feature')


## interactions and polynomials

# interaction and polynomial featrures enrich feature representation. often used in statistical modeling

# what is the difference between statistical modeling and machine learning???

### add binned feature
X_combined = np.hstack([X, X_binned])
X_combined.shape

reg = LinearRegression().fit(X_combined, y)

# there is a single x-axis feature, so all the bins have the same slope?
line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label='linear regression combined')
# equal to: plt.vlines(bins, -3, 3)
for bin in bins:
	plt.plot([bin, bin], [-3, 3], ':', c='k')
plt.legend(loc='best')
plt.ylabel('regression output')
plt.xlabel('input feature')
plt.plot(X, y, 'o', c='k')


### add interaction
X_product = np.hstack([X_binned, X * X_binned])
X_product.shape

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label='linear regression product')
for bin in bins:
	plt.plot([bin, bin], [-3, 3], ':', c='k')
plt.plot(X, y, 'o', c='k')
plt.ylabel('regression output')
plt.xlabel('input feature')
plt.legend(loc='best')


### add polynomials
from sklearn.preprocessing import PolynomialFeatures

# degree=10: polynomials up to x ** 10
# include_bias=True: add a feature that's constantly 1
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

X_poly.shape
X[:5]
X_poly[:5]
poly.get_feature_names()


reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
plt.plot(X, y, 'o', c='k')
plt.ylabel('regression output')
plt.xlabel('input feature')
plt.legend(loc='best')


# more complex model, say kernel SVM, will get similar predictions as polynomial regression without explicit transformation of the features 
from sklearn.svm import SVR

for gamma in [1, 10]:
	svr=SVR(gamma=gamma).fit(X, y)
	plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))

plt.plot(X, y, 'o', c='k')
plt.ylabel('regression output')
plt.xlabel('input feature')
plt.legend(loc='best')



from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

# scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# transform based on the scaler of X_train
X_test_scaled = scaler.transform(X_test)


poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
X_train.shape
X_train_poly.shape

# PolynomialFeatures has parameter include_bias, its value is True by default, which adds a feature that's constantly 1
poly.get_feature_names()
len(poly.get_feature_names())



## univariate nonlinear transformation

# log exp sin cos etc. can help by adjusting the relative scales in the data

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
y
print('number of integer values appearances for the first feature of X(poisson distribution):\n{}'.format(np.bincount(X[:,0])))
min(X[:,0])
max(X[:,0])

# bar plot
bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color='b')
plt.ylabel('number of appearances')
plt.xlabel('value')

# ridge regression on the original X y
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
score

# use log transformation
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
plt.hist(np.log(X_train_log[:, 0] + 1), bins=25, color='gray')
plt.ylabel('number of appearances')
plt.xlabel('value')
score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
score



## automatic feature selection

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
X_w_noise.shape

# test_size, if float, represent the propotion of the test set. if int, represent the number of the test set
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)
len(X_w_noise)
len(X_train)
len(y_test)

# with f_classif (default relationship test between feature and target label in classification), select 50% features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# transform train data
X_train_selected = select.transform(X_train)

X_train.shape
X_train_selected.shape

# get_support get the features selected
mask = select.get_support()
mask
# shape of 1 * x （-1 means the length of original dataset
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('sample index')



from sklearn.linear_model import LogisticRegression
# transform test datasets
X_test_selected = select.transform(X_test)
# build a logistic model
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

# which features are selected
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('sample index')

# score on test before feature selection
LogisticRegression().fit(X_train, y_train).score(X_test, y_test)
X_test_l1 = select.transform(X_test)
# score on test data after model-based feature selection
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
score

### iterative selection

# supervised method, need the target label, split the data into train and test
# a series models are built with varying numbers of features. two methods, 1) add 1 feature at a time until some stopping criterion is reached. 2) eliminate 1 feature at a time with all features until some stopping criterion is reached

# RFE(recursive feature elimination), start with all features, using a specific model, discard one with least importance, and repeat until prespecified number of features are left
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)
select.fit(X_train, y_train)

# which features are selected
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('sample index')

# score on test before feature selection
LogisticRegression().fit(X_train, y_train).score(X_test, y_test)
# select features
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
# score on test data after iterative feature selection
score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
score

# use the model used inside the RFE to predict on test data which will only use the selected features
select.score(X_test, y_test)



## utilizing expert konwleadge

# prior knowledge of the data which cannot be captured by models from initial representation of data

# load data citibike
citibike = mglearn.datasets.load_citibike()

# first impression of the data
citibike.head()
plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
# ha, xticks' position
plt.xticks(xticks, xticks.strftime('% a %m-%d'), rotation=90, ha='left')
plt.plot(citibike, linewidth=1)
plt.xlabel('Date')
plt.ylabel('Rentals')

y = citibike.values

# convert time to POSIX(the number of seconds since January 1970 00:00:00, aka(as known as) the beginning of Unix time) type, dividing by 10**9
X = citibike.index.astype('int64').values.reshape(-1, 1) // 10**9
X.shape

# use the first 184 data as the training
n_train = 184 

# def a function, evaluate with specific regressor, then plot the training test y and the predicted y
def eval_on_features(features, target, regressor):
	# split the data to training and test not randomly
	X_train, X_test = features[:n_train], features[n_train:]
	y_train, y_test = target[:n_train], target[n_train:]
	regressor.fit(X_train, y_train)
	print('test-set R^2: {:.2f}'.format(regressor.score(X_test, y_test)))
	y_pred = regressor.predict(X_test)
	y_pred_train = regressor.predict(X_train)
	
	plt.figure(figsize=(10, 3))
	# ?
	plt.xticks(range(0, len(x), 8), xticks.strftime('%a %m-%d'), rotation=90, ha='left')
	# 横轴第一区间，y_train
	plt.plot(range(n_train), y_train, label='train')
	# 轴第一区间，y_pred_train
	plt.plot(range(n_train), y_pred_train, '--', label='prediction train')
	# 横轴第二区间， y_test
	plt.plot(range(n_train, len(y_test)+n_train), y_test, '-', label='test')
	# 横轴第二区间，y_pred
	plt.plot(range(n_train, len(y_test)+n_train), y_pred, '--', label='prediction test')
	plt.legend(loc=(1.01, 0))
	plt.xlabel('Date')
	plt.ylabel('Rentals')


from sklearn.ensemble import RandomForestRegressor
# use RandomForestRegressor to predct, the result is not good, because the value of the POSIX time for the test set is outside of the range in the training set, trees cannot extrapolate to feature range outside the training set
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# plt.figure()
eval_on_features(X, y, regressor)

# expert knowledge: the time of day, the day of the week

# reshape(-1) 1 row, unknown columns
# reshape(-1, 1) unknown rows, 1 column
# reshape(-1, 3) unknown rows, 3 columns
# reshape(2, -1) 2 rows, unknown columns

# the time of day
# citibike.index is Int64Index object, which has no arrtibute reshpe. it has to be transformed to values
X_hour = citibike.index.hour.values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)

# the time of day and the day of the week
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1), citibike.index.hour.values.reshape(-1, 1)])
eval_on_features(X_hour_week, y, regressor)

# hour and dayofweek are categorical, not suit for linear model
from sklearn.linear_model import LinearRegression
eval_on_features(X_hour_week, y, LinearRegression())

# use one hot encode to transform categorical data to dummy variables
encoder = OneHotEncoder()
X_hour_week_onehot = encoder.fit_transform(X_hour_week).toarray()
from sklearn.linear_model import Ridge
eval_on_features(X_hour_week_onehot, y, Ridge())

# add polynomial features
poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)


hour = ['%02d:00' % i for i in range(0, 24, 3)]
day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
features = day + hour

# get_feature_names name, all the interaction features extracted by PolynomialFeatures
features_poly = poly_transformer.get_feature_names(features)
# lr, ridge, fit on X_hour_week_onehot_poly
# get the features with coefficients are nonzero
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]
# plot nonzero coef of Ridge model, and the interaction features
plt.figure(figsize=(15, 2));
plt.plot(coef_nonzero, 'o');
# set the locations, labels of the xticks, rotation angle
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90);
plt.xlabel('Feature names');
plt.ylabel('Feature magnitude');
