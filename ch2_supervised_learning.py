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

# both LogisticRegression and LinearSVC apply an L2 regularization, the trade-off parameter which determines the strength of the regularization is C(default=1), lower C corresponding to more regularization with the coefficients(w) close to zero, in other words the model is more general

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
# MultinomialNB, count data

# BernoulliNB, for binary data, it counts how often every feature of of each lass is not zero, here is an exmaple

X = np.array([[0, 1, 0, 1],
			  [1, 0, 1, 1],
			  [0, 0, 0, 1],
			  [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
	# iterate over each class
	# count (sum) entried of 1 per feature
	counts[label] = X[y == label].sum(axis=0)
print('feature counts:\n{}'.format(counts))

# to predict, a data point is compared to the statistics for each of the classes, and the best matching class is predicted
# MultinomialNB and BernoulliNB have a parameter alpha, large alpha means more smoothing, less complex model


### Decision Tree
# decision tree is like a series of yes or no questions
# an illustration of decision tree
mglearn.plots.plot_animal_tree()

# learning a decision tree means learning the sequence of if/else questions. the questions are called test. the tests on continuous data are of the form "is feature i larger than value a". the tests will span to a tree, when a new point arrives, it will flow a test path untill leads to a leaf

 # there are two ways avoid overfitting, 1.stop creating the tree early(pre_pruning), 2.built a whole tree, then remove nodes that contain little information(post-pruning)
 # scikit-learn only implements pre-pruning
from sklearn.tree import DecisionTreeClassifier
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# random_state, used for tie-breaking internally
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
tree.score(X_train, y_train)
tree.score(X_test, y_test)


# parameter max_depth, pre-prune the tree with a certain depth
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
tree.score(X_train, y_train)
tree.score(X_test, y_test)

#### analyzing decision trees
# export_graphviz function visualize the tree, generate a .dot file
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='cancer.tree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names, impurity=False, filled=True)

# read .dot file to see the visulization
import graphviz

with open('cancer.tree.dot') as f:
	dot_graph = f.read()
graphviz.Source(dot_graph)



#### featrue importance in trees
# featrue importance rates how important each featrue is for the decision a tree makes. it is a number between 0 and 1 for each featrue(0 means not used, 1 means perfectly predict the target). the featrue importances sum to 1
tree.featrue_importances_


def plot_feature_importances_cancer(model):
	# shape[1] for number of featrues, shape[0] for number of
	# cancer.data.shape?
	n_featrues = cancer.data.shape[1]
	plt.barh(range(n_featrues), model.featrue_importances_, align='center')
	# range(10)
	# np.arange(10)?
	plt.yticks(np.arange(n_featrues), cancer.feature_names)
	plt.xlabel('featrue importance')
	plt.ylabel('featrue')

plot_feature_importances_cancer(tree)

# a featrue with low featrue_importance may be uninformtive or not picked by this tree for another featrue encodes the same information
# featrue importances are always positive. uslike regression coefficients may interpretable. a featrue with high importance indicates this featrue is important, but will not tell a sample should be classified to A or B


# the x featrue has 0 featrue_importance, the y featrue has 1 featrue_importance. y has a nonmomotonous relationship with the class label
tree = mglearn.plots.plot_tree_not_monotone()
# display?
display(tree)



#### decision tree regression
# all tree_based regression models(include DecisionTreeRegressor) is not able to etrapolate, or make predictions outside of the range of the training data

# prepare the data set?
ram_prices = pd.read_csv('ram_price.csv')
ram_prices
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel('year')
plt.ylabel('price in $/Megabyte')

from sklearn.tree import DecisionTreeRegressor
# use historical data to forecast prices after the year 2000
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# np.newaxis?
X_train = data_train.date[:, np.newaxis]
# logarithm of y
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

# predict on all data
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# undo logarithm
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

# semilogy?
plt.semilogy(data_train.date, data_train.price, label='training data')
plt.semilogy(data_test.date, data_test.price, label='test data')
# tree regression cannot predict outside the training data set
plt.semilogy(ram_prices.date, price_tree, label='tree prediction')
plt.semilogy(ram_prices.date, price_lr, label='linear prediction')
plt.legend()

# decision tree can be easily visualized. the splits of data don't depend on scaling or preprocessing of data, but it tend to be overfitting even with pre_pruning

### Ensembles of Decision Trees
#### random forests
# a random forest is a collection of decision trees, where each tree is slightly different from the others. we can average the results of different trees to avoid overfitting
# the trees in a random forest are ranomized, 1. selecting the data points randomly to build a tree, 2. selecting the features randomly in each split test


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# parameter n_etimators, the number of trees to build
# parameter n_samples, number of repeat times in bootstrap sampling, which will creat a dataset as big as the original one to make a decision tree
# parameter max_features, the number of featrues selected in a decision tree, small max_features reduces overfitting. a good rule of thumb to select max_features: sqrt(n_featrues) for classification, log2(n_featrues) for regression
# parameter random_state? different trees

# in prediction, random forest average the results in regression or use soft voting in classification
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
# forest.estimators_ ?
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
	ax.set_title('tree {}'.format(i))
	# the partition plot of every tree in forest
	mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
# ?
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax= axes[-1, -1], alpha=.4)
axes[-1, -1].set_title('random forest')
# ?
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
forest.score(X_train, y_train)
forest.score(X_test, y_test)

plot_feature_importances_cancer(forest)


# random forests do not perform well on high dimensional, sparse data such as text data(linear model might be appropriate)
# random forest require more memory and are slower to train and predict than linear models



#### Gradient boosting machines(gradient boosted regression trees)
# it builds trees in a serial manner, where each tree tries to correct the mistakes of the previous one. it often uses strong pre_pruning to build shallow trees, of depth 1 to 5, that uses less memory and predicts faster
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
gbrt.score(X_train, y_train)
gbrt.score(X_test, y_test)

# lower learning_rate can reduce overfitting
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
gbrt.score(X_train, y_train)
gbrt.score(X_test, y_test)

# fewer tree depth can reduce overfitting
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
gbrt.score(X_train, y_train)
gbrt.score(X_test, y_test)


plot_feature_importances_cancer(gbrt)

# for large-scale dataset, package xgboost maybe more powerful in building a gradient boosting model


### Kernelized Support Vector Machines

X, y = make_blobs(centers=4, random_state=8)
y
# make y as categorical variable 
y = y % 2

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('featrue 0')
plt.ylabel('featrue 1')

# build a linear svm model
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)

# 2d separator, a line
mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('featrue 0')
plt.ylabel('featrue 1')

X.shape
# add the squared feature 1
X_new = np.hstack(X, X[:, 1:] ** 2)

from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
# visualize in 3D
ax = Axes3D(figure, elev=-152, azim=-26)
# plot all the points with y == 0,
# plot all the points with y == 1
mask = y == 0

# X_new[mask, 2] the added new featrue
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
ax.set_xlabel('featrue 0')
ax.set_ylabel('featrue 1')
ax.set_zlabel('featrue1 ** 2')


linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
coef
intercept

# linear decision boundry
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 2].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
# decision boundry(a plane)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]

ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)

ax.set_xlabel('featrue 0')
ax.set_ylabel('featrue 1')
ax.set_zlabel('featrue 1 ** 2')


ZZ = YY ** 2
# decision_function?
# np.c_?
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('featrue 0')
plt.ylabel('featrue 1')


#### kernel SVMs
# kernel trick works by directly computing the distace of the data points for the expanded feature representation

# polynomial kernel, computes all possible polynomials up to a certain degree of the original feature
# RBF(radial basis function), also known as the Gaussian kernel, it considers all possible polynomials of all degrees, but the importance of the featrues decreases for higher degrees

# name of SVM for classification, only a subset of the training points matter for defining the decision boundry: the ones lie on the border between the classes. these are called support vectors


# to make a prediction for a new point, the distance to each of the support vectors is measured, the importance of the support vectors is stored in the dual_coef_ attribute of SVC, a classification decision is made based on the distances to the support vector



from sklearn.svm import SVC
# SVC is nonlinear
# from sklearn.svm import LinearSVC
#  this is a linear svm

X, y = mglearn.tools.make_handcrafted_dataset()
# train a kernel SVM. parameter gamma controls the width of the Gaussian kernel. it determines the scale for points how close together. parameter C is a regularization parameter. it limits the importance of each support vector(value of support vector in dual_coef_)
SVC?
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
# plot the decision boundry
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
# scatter plot
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# support vectors
sv = svm.support_vectors_

sv_labels = svm.dual_coef_.ravel() > 0
# plot support vectors with their importance
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel('featrue 0')
plt.ylabel('featrue 1')




##### tuning SVM parameters
# small gamma indicates a large radius for Gaussian kernel, means points are considered close by, make the model more general
# large C makes the model less regularization
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, [-1, 0, 3]):
	for a, gamma in zip(ax, range(-1, 2)):
		mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
# why the legend's position there?
axes[0, 0].legend(['class 0', 'class 1', 'sv class 0', 'sv class 1'], ncol=4, loc=(.9, 1.2))


# apply svc to breast cancer datasets
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# by default, SVC uses rbf kernel with C=1, gamma=1/n_featrues
svc = SVC()
svc.fit(X_train, y_train)

svc.score(X_train, y_train)
svc.score(X_test, y_test)


# see the scale of each featrue
X.shape
X_train.min(axis=0)
plt.plot(X_train.min(axis=0), 'o', label='min')
plt.plot(X_train.max(axis=0), '^', label='max')
plt.legend(loc=4)
plt.xlabel('featrue index')
plt.ylabel('featrue magnitude')
# y axis in log scale
plt.yscale('log')



#### preprocessing data for SVMs
# minimum value per feature
min_on_training = X_train.min(axis=0)
# range of each feature(max - min)
range_on_training = (X_train - min_on_training).max(axis=0)
# scale the training data to 0-1
X_train_scaled = (X_train - min_on_training) / range_on_training
X_train_scaled.max(axis=0)
X_train_scaled.min(axis=0)

# use the same transformation on test set
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC()
svc.fit(X_train_scaled, y_train)
svc.score(X_train_scaled, y_train)
svc.score(X_test_scaled, y_test)


svc1000 = SVC(C=1000)
svc1000.fit(X_train_scaled, y_train)
svc1000.score(X_train_scaled, y_train)
svc1000.score(X_test_scaled, y_test)



### Neural Networks(Deep Learning)
#### the neural network model
# multilayer perceptrons(MLPs) are known as (vanilla) feed-forward neural networks
# MLPs can be viewed as generalizations of linear models that perform multiple stages of preprocessing to come to a decision

# a graph illustration of linear model
display(mglearn.plots.plot_logistic_regression_graph())

# graph illustrations of MLPs
display(mglearn.plots.plot_single_hidden_layer_graph())
mglearn.plots.plot_two_hidden_layer_graph()

line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label='tanh')
plt.plot(line, np.maximum(line, 0), label='relu')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('relu(x), tanh(x)')


#### Tuning nerural networks
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

mlp = MLPClassifier(algorithm='l-bfgs', random_state=0).fit(X_train, y_train)
# decision boundry
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
# scatter plot
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel('feature 0')
plt.ylabel('feature 1')


# mlp model with 10(100, by default)hidden nodes and 1 layer
mlp = MLPClassifier(algorithm='l-bfgs', random_state=0, hidden_layer_sizes=[10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel('feature 0')
plt.ylabel('feature 1')

# using two hidden layers, with 10 nodes each
mlp = MLPClassifier(algorithm='l-bfgs', random_state=0, hidden_layer_sizes=[10, 10])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel('featrue 0')
plt.ylabel('featrue 1')

# two hidden layers, 10 nodes each, with tanh nonlinearity(default, relu-rectifying nonlinearity, or rectified linear unit)
mlp = MLPClassifier(algorithm='l-bfgs', activation='tanh', random_state=0, hidden_layer_sizes=[10, 10])
mpl.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.xlabel('featrue 0')
plt.ylabel('featrue 1')

# mlp uses an l2 penalty to shrink the weights toward zero
# parameter alpha in mlp controls the model complexity, large alpha makes it more general
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
	for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
		mlp = MLPClassifier(algorithm='l-bfgs', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
		mlp.fit(X_train, y_train)
		mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3, ax=ax)
		mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
		ax.set_title('n_hidden_nodes=[{}, {}]\nalpha={:.4f}'.format(n_hidden_nodes, n_hidden_nodes, alpha))

# parameter random_state in mlp, set initial weights randomly, which will affect the model
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
	mlp = MLPClassifier(algorithm='l-bfgs', random_state=i, hidden_layer_sizes=[100, 100])
	mlp.fit(X_train, y_train)
	mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3, ax=ax)
	mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
	ax.set_title('n_hidden_nodes=[100, 100]\n random_state={}'.format(i))


# apply MLPClassifier to the breast cancer datsets
from sklearn.datsets import load_breast_cancer
cancer = load_breast_cancer()

print('cancer data per-feature maxima:\n{}'.format(cancer.data.max(axis=0)))

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

mlp.score(X_train, y_train)
mlp.score(X_test, y_test)


# it's better to scale the original data before applying mlp
# mean value of per feature on the training data
mean_on_train = X_train.mean(axis=0)
# standard deviation of per feature
std_on_train = X_train.std(axis=0)
# scale the training data to be 0 mean and 1 std
X_train_scaled = (X_train - mean_on_train) / std_on_train
# scale the test data the same way using train mean and std
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)

mlp.score(X_train_scaled, y_train)
mlp.score(X_test_scaled, y_test)


# compare with SVM classifier
from sklearn.svm import SVC
svc = SVC().fit(X_train_scaled, y_train)
svc.score(X_train_scaled, y_train)
svc.score(X_test_scaled, y_test)

# increase the number of iterations
# about algorithm adam?
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)
mlp.score(X_train_scaled, y_train)
mlp.score(X_test_scaled, y_test)

# increase alpha makes the model more general, alpha=0.0001 by default
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)
mlp.score(X_train_scaled, y_train)
mlp.score(X_test_scaled, y_test)

plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('columns in weight matrix')
plt.ylabel('input feature')
plt.colorbar()

### uncertainty estimates from classifiers

# you are not only interested in which class a classifier predicts for a certain test point, but also how certain it is that this is the right class

# scikit-learn obtains uncertainty estimates from classifiers using decision_function and predict_proba

from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.ensemble make_blobs, make_circles
X, y = make_circles(noise=0.25, factor=o.5, random_state=1)
X
y

# rename the y label using blue and red
y_named = np.array(['blue', 'red'])[y]

X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)


gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

# model method decision_function has shape (n_samples,)
gbrt.decision_function(X_test)
X_test.shape
gbrt.decision_function(X_test).shape

# for binary classification, the negative class is the first entry of the classes_ attribute
gbrt.decision_function(X_test) > 0
gbrt.predict(X_test)


# make true/false into 0 and 1
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
pred = gbrt.classes_[greater_zero]

pred
gbrt.predict(X_test)
# these two are the same
np.all(pred == gbrt.predict(X_test))



































