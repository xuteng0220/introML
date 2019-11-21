# Supervised Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

## Classification and Regression
# distinguish between classfication and regression is to ask whether there is some kind of continuity in the output


## supervised machine learning algorithm

### some sample datasets
X, y = mglearn.datasets.make_forge()
X
y
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['class 0', 'class 1'], loc=4)
plt.xlabel('first feature')
plt.ylabel('second feature')


X, y = mglearn.datasets.make_wave(n_samples=40)
X
y
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('feature')
plt.ylabel('target')


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer
cancer.keys()
cancer.data.shape
# np.bincount?
print('sample counts per class:\n{}'.format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
cancer.feature_names


from sklearn.datasets import load_boston
boston = load_boston()
boston.data.shape
X, y = mglearn.datasets.load_extended_boston()
X.shape


### k-Nearest Neighbors
#### k-Nearest classification
# a point to be predicted, first find the k nearest neighbors of it, then take a vote on the class of these neighbors, the largest voting class is the one to be predict
# illustration of k-Nearest neighbors
mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)


#### decision boundary of k-Nearest classification
# decision boundary is the divide between where the algorithm assigns class 0 versus where it assigns class 1
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
	clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
	# mglearn.plots.plot_2d_separator? decision boundary
	mglearn.plots.plot_2d_separator(clf, x, fill=True, eps=0.5, ax=ax, alpha=.4)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	ax.set_title('{} neighor(s)'.format(n_neighbors))
	ax.set_xlabel('feature 0')
	ax.set_ylabel('feature 1')
axes[0].legend(loc=3)



from sklearn.datasets import load_breast_cancer
 cancer = load_breast_cancer()
 # stratify?
 X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

neighbors_setting = range(1, 11)

for n_neighbors in neighbors_setting:
	clf = KNeighborsClassifier(n_neighbors = n_neighbors)
	clf.fit(X_train, y_train)
	training_accuracy.append(clf.score(X_train, y_train))
	test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_setting, training_accuracy, label='training accuracy')
plt.plot(neighbors_setting, test_accuracy, label='test accuracy')
plt.ylabel('accuracy')
plt.xlabel('n_neighbors')
plt.legend()


#### k neighbors regression
# illustration of k neighbors regression
mglearn.plots.plot_knn_regression(n_neighbors=1)
mglearn.plots.plot_knn_regression(n_neighbors=3)


from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
reg.predict(X_test)
the score of k neighbors regression is the R^2 
print('test set R^2: {:.2f}'.format(reg.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# n rows, 1 column
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
	reg = KNeighborsRegressor(n_neighbors=n_neighbors)
	reg.fit(X_train, y_train)
	ax.plot(line, reg.predict(line))
	ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
	ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1)m markersize=8)
	ax.set_title('{} neighbor(s)\n train score: {:.2f} test score: {:.2f}'.format(n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
	ax.set_xlabel('feature')
	ax.set_ylabel('target')
axes[0].legend(['model prediction', 'training data/target', 'test data/target'], loc='load_breast_cancer')

# weakness of kNN, when the training set is vaer large (with many features, hundreds or more), the prediction can be slow. with many sparse features, it dose not perform well


### linear models
$y^\hat = \beta_0x_0 + \beta_1x_1 + \dots + \beta_px_p + b$

mglearn.plots.plot_linear_regression_wave()
# for datasets  with many features, linear models can be powerful. In particular, if the features are more than tranining data points, any target y can be perfectly modeled as a linear function. why? feature, n varianles, training points, m equations, n > m, solvable


#### linear regression
# linear regression has no parameterd, also has no way to control model complexity
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

# sklearn always stores anything derived from the training data that end with a trailing underscore
lr.coef_
lr.intercept_

lr.score(X_train, y_train)
lr.score(X_test, y_test)

# linear regression on more complex datasets
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
lr.score(X_train, y_train)
lr.score(X_test, y_test)



#### ridge regression
# ridge regression select coefficients(w), 1.fit well for training set, 2.content with an additional constraint that the magnitude of coefficients is as small as possible, all entries of w should be close to zero. intuitively, each feature has little effect on the outcome
# ridge penalizes the L2 norm of the coefficients, or the Euclidean length of w, why?
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
ridge.score(X_train, y_train)
ridge.score(X_test, y_test)
# try?
ridge.coef_
rideg.intercept_

# Ridge's parameter alpha, increasing alpha forces coefficients to move toward zero, which decreases traing set performance but help generalization(trade-off between model simplicity and performance on training set)

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge10.score(X_train, y_train)
ridge10.score(X_test, y_test)

# small alpha leads the Ridge to LinearRegression
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
ridge01.score(X_train, y_train)
ridge01.score(X_test, y_test)



plt.plot(ridge.coef_, 's', label='Ridge alpha=1')
plt.plot(ridge10.coef_, '^', label='Ridge alpha=10')
plt.plot(ridge01.coef_, 'v', label='Ridge alpha=0.1')
plt.plot(lr.coef_, 'o', label='LinearRegression')
plt.xlabel('coefficients index')
plt.ylabel('coefficients magnitude')
# hlines?
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

# LinearRegression vs Ridge about performance on training set and its generalization effects(performance on test set) arrording to the size of training set
mglearn.plots.plot_ridge_n_samples()



#### Lasso
# Lasso restricts coefficients to be close to zero with L1 regularization, some coefficients are exactly zero(some entries are ignored by the model, can be seen as a form of automatic feature selection)

from sklearn.linear_model import Lasso

# entended boston data set
lasso = Lasso().fit(X_train, y_train)
lasso.score(X_train, y_train)
lasso.score(X_test, y_test)
# number of features used
np.sum(lasso.coef_ != 0)


# large alpha(default 1) makes coefficients close to zero, which means the model is more generalized(cause underfitting). decreasing alpha, should increase the max_iter parameter(the maximum number of iterations to run)
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
lasso001.score(X_train, y_train)
lasso001.score(X_test, y_test)
np.sum(lasso001.coef_ != 0)


lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
lasso00001.score(X_train, y_train)
lasso00001.score(X_test, y_test)
np.sum(lasso00001.coef_ != 0)


plt.plot(lasso.coef_, 's', label='lasso alpha=1')
plt.plot(lasso001.coef_, '^', label='lasso alpha=0.01')
plt.plot(lasso00001.coef_, 'v', label='lasso alpha=0.00001')
plt.plot(ridge01.coef_, 'o', label='ridge alpha=0.1')
# ncol?
plt.legend(ncol=2, loc=(0, 1.05))
# figsize?
plt.ylim(-25, 25)
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')




### linear models for classification

$y^\hat = \beta_0x_0 + \beta_1x_1 + \dots + \beta_px_p + b > 0$

# both LogisticRegression and LinearSVC apply an L2 regularization, the trade-off parameter which determines the strength of the regularization is C(default=1), higher C corresponding to less regularization with the coefficients(w) close to zero

from sklearn.linear_model import LogisticRegression
# in sklearn svm stands for linear support vector machines, which include svc(linear support classifier)
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# try list(zip([1, 2], [3, 4]))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
	clf = model.fit(X, y)
	# plot_2d_separator, plot the decision boundry
	mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
	mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	ax.set_title('{}'.format(clf.__class__.__name__))
	ax.set_xlabel('feature 0')
	ax.set_ylabel('feature 1')
axes[0].legend()

# an illustration of LinearSVC with different trade-off parameter C
mglearn.plots.plot_linear_svc_regularization()


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
logreg.score(X_train, y_train)
logreg.score(X_test, y_test)


logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
logreg100.score(X_train, y_train)
logreg100.score(X_test, y_test)

logreg001 = LinearRegression(C=0.01).fit(X_train, y_train)
logreg001.score(X_train, y_train)
logreg001.score(X_test, y_test)

# different parameter C leads to some specific feature coefficients to be positive or negative, so it illustrates that interpretations of coefficients of linear model should be taken with a grain of salt
plt.plot(logreg.coef_.T, 'o', label='default C=1')
plt.plot(logreg100.coef_.T, '^', label='C=100')
plt.plot(logreg001.coef_.T, 'v', label='C=0.01')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel('coefficients index')
plt.ylabel('coefficients magnitude')
plt.legend()



for C, marker in zip([0.0001, 1, 100], ['o', '^', 'v']):
	# LogisticRegression with L1 norm regularization
	lr_l1 = LogisticRegression(C=C, penalty='l1').fit(X_train, y_train)
	print('training accuracy of l1 logreg with C={:.3f}: {:.2f}').format(C, lr_l1.score(X_train, y_train))
	print('test accuracy of l1 logreg with C={:.3f}: {:.2f}').format(C, lr_l1.score(X_test, y_test))
	plt.plot(lr_l1.coef_.T, marker, label='C={:.3f}'.format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')
plt.ylim(-5, 5)
plt.legend(loc=3)


### linear model for multiclass classfication
# multiclass classification, one-vs-rest approach, 3 classes classification will have 3 binary classifier models. to predict, all binary classifiers are run on a unlabeled point. the classifier that has the highest score on its single class wins.
from sklearn.datsets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.legend(['class 0', 'class 1', 'class 2'])

# a linear support vector classifier
linear_svm = LinearSVC().fit(X, y)
linear_svm.coef_.shape
linear_svm.intercept_.shape

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# why the axes are not from -15 to 15?
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
	# classification line
	plt.plot(line, -(line * coef[0] + intercept) coef[1], c=color)
# ylim
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.legend(['class0', 'class1', 'class2', 'line class0', 'line class1', 'line class2'], loc=(1.01, 0.3))

# decision boundry
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
	plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.legend(['class0', 'class1', 'class2', 'line class0', 'line class1', 'line class2'], loc=(1.01, 0.3))

# what is the result?
linear_svm.predict(x[1, 0])




# linear model has a main parameter, alpha in regression(linear regression, Ridge, Lasso), C in classification(LinearSVC, LogisticRegression), large alpha or small C mean simple models(more generalized)
# L1 regularization results in fewer features used in model than L2, and also is easier to explain the model



### Naive Bayes classifiers
# help(Naive Bayes)?
# Naive Bayes is fast in training
# GaussianNB, continuous data
# BernoulliNB, binary data
# MultinomialNB, count data

X = np.array([[0, 1, 0, 1],
			  [1, 0, 1, 1]
			  [0, 0, 0, 1]
			  [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in unique(y):
	# iterate over each class
	# count (sum) entried of 1 per feature
	counts[label] = X[y == label].sum(axis=0)
print('feature counts:\n{}'.format(counts))
