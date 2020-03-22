# Model evaluation and improvement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

%matplotlib inline

# an example
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# create a synthetic dataset
X, y = make_blobs(random_state=0)
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression().fit(X_train, y_train)
logreg.score(X_test, y_test)


## cross-validation
# cross validation evaluate generalization performance of a model, which is more stable and thorough than s dplit into a training and a test set.
# the data is split repeatedly and multiple models are trained
# most common cross validation is k-fold cross-validation
# when you call cross-validation, it won't return a model. It only aims at the generalization of a given model.

mglearn.plots.plot_cross_validation()


from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model imp 	LogisticRegression

iris = load_iris()
logreg = LogisticRegression()
# cross_val_score perform three fold by default, return three accurate values
scores = cross_val_score(logreg, iris.data, iris.target)
scores

# perform 5-fold 
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
scores


### Stratified k-fold cross-validation and othe strategies

# in stratified cross-validation, the data is split theat the proportions between classes are the same in each fold as they are in the whole dataset
mglearn.plots.plot_stratified_cross_validation()


from sklearn.datasets import load_iris
iris = load_iris()
# here iris has two classes, say, X:Y = 90% : 10%. then, when using stratified cross-validation(suppose 3 folds, a, b, c), each fold has 90% from class X and 10% from class Y
iris.target
?np.array(iris.target).value_count


# to evaluate a classifier, it's more reliable using stratified k-fold cross-validation than standard k-fold cross-validation
# for regression, use standard k-fold cross-validation is default


from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
# cv=kfold split the data fixed when you call it repeatedly, cv=5 split the data randomly everytime you call it 
cross_val_score(logreg, iris.data, iris.target, cv=kfold)

# return??
kfold.split(iris.data, iris.target)


# 1st call kfold=5, different ???
cross_val_score(logreg, iris.data, iris.target, cv=5)
# 2nd call kfold=5
cross_val_score(logreg, iris.data, iris.target, cv=5)


# shuffle=True  split the data in a shuffle way
# random_state=0 fix a random seed
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
cross_val_score(logreg, iris.data, iris.target, cv=kfold)


### leave one out cross-validation
# for each split, you pick a single data to be the test set, and it will repeate n time(n: number of the data)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
len(iris.target)
len(scores)
scores.mean()

# ?
scores = cross_val_score(logreg, iris.data, iris.target, cv=1)


### shuffle-split cross-validation
mglearn.plots.plot_shuffle_split()

from sklearn.model_selection import ShuffleSplit
# test_size fraction(or absolute number) of the test size. n_splits number of split time
# test_size+train_size could less than 1
shuffle_split = ShuffleSplit(test_size=0.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
scores

# stratified shuffle split
from sklearn.model_selection import StratifiedShuffleSplit
stratified_shuffle_split = StratifiedShuffleSplit(test_size=0.3, train_size=0.6, n_splits=3)
cross_val_score(logreg, iris.data, iris.target, cv=stratified_shuffle_split)


## Cross-validation with groups
# each group is either entirely in the training set or entirely in the test set
# data in the same group may have higher correlation
# data from the same group are in both training and test set may worse the generalization of model
mglearn.plots.plot_label_kfold()


from sklearn.model_selection import GroupKFold
# create synthetic dataset
X, y = make_blobs(n_samples=12, random_state=0)
# first 3 samples belong to group 1, etc.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
scores


## Grid search
# try all possible combinations of the parameters in a model

### simple grid search
# naive grid search implementation
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
X_train.shape
X_test.shape

best_score = 0

# grid search use for loop
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
	for C in [0.001, 0.01, 0.1, 1, 10, 100]:
		# gamma, kernel bandwidth, C regularization parameter
		svm = SVC(gamma=gamma, C=C)
		svm.fit(X_train, y_train)
		score = svm.score(X_test, y_test)

		if score > best_score:
			best_score = score
			best_parameters = {'C': C, 'gamma':gamma}

best_score
best_parameters

# parameters selected above may not be the real best one, because the test set is used to selecte parameters. we donot know whether the selected ones are good at generalization to new data.
# so we have to split the data to train(train the model), cross validation(selecte parameters) and test(test generalization) set


mglearn.plits.plot_threefold_split()


from sklearn.svm import SVC
# split data into training validation and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
X_train.shape
X_valid.shape
X_test.shape

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
	for C in [0.001, 0.01, 0.1, 1, 10, 100]:
		svm = SVC(gamma=gamma, C=C)
		svm.fit(X_train, y_train)

		score = svm.score(X_valid, y_valid)
		if score > best_score:
			best_score = score
			best_parameters = {'gamma':gamma, 'C':C}


# rebuild the model with the selected parameters
svm = SVC(**best_parameters)
# fit the model with train+valid set, use as much data as possible
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print('best score on validation set: {:.2f}').format(best_score)
print('best parameters:', best_parameters)
print('test score with best parameters: {.2f}'.format(test_score))




### grid search with cross validation

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
	for C in [0.001, 0.01, 0.1, 1, 10, 100]:
		svm = SVC(gamma=gamma, C=C)
		# perform cross-validation
		scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
		# compute mean cross-validation accuracy
		score = np.mean(scores)

		if score > best_score:
			best_score = score
			best_parameters = {'C':C, 'gamma':gamma}

# rebuild svm model on conbined train and validation set
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)

print('score on cross-validation: {:.2f}'.format(best_score))
print('best parameters selected by cross-validation:', best_parameters)
print('score on test set: {:.2f}'.format(svm.score(X_test, y_test)))



# an illustration of how the best parameters are selected
mglearn.plots.plot_cross_val_selection()


# cross-validation is often used in conjunction with parameters seach methods like grid search

# the overall process of splitting the data, running the grid search, and evaluating the final parameters
mglearn.plots.plot_grid_search_overview()



### GridSearchCV class
# GridSearchCV class implements the grid search for parameters selection
# first, specify the parameters to be searched using a dictionary
# sencond, instantiate the GridSearchCV class
param-grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
# GridSearchCV 1.selecte the best parameters using cross-validation on training set (split the training set into training and validation set)
# 2.fit the whole training set using selected parameters
grid_search.fit(X_train, y_train)
# best parameters selected by grid_search
grid_search.best_params_
# best_score_ attribute stores the mean cross-validation accuracy with cross-validation performed on the training set
grid_search.best_score_
print('score on test set {:.2f}'.format(grid_search.score(X_test, y_test)))



# cv_results_ attribute, is a dictionary storing all aspects of grid search
results = pd.DataFrame(grid_search.cv_results_)
results.head()

# plot the mean cross-validation scores
scores = np.array(results.mean_test_score).reshape(6, 6)

# heatmap, light color means high accuracy
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'], ylabel='C', yticklabels=param_grid['C'], cmap='viridis')



fig, axes = plt.subplotd(1, 3, figsize=(13, 5))

param_grid_linear = {'C': np.linspace(1, 2, 6), 'gamma': linspace(1, 2, 6)}

param_grid_one_log = {'C': np.linspace(1, 2, 6), 'gamma': logspace(-3, 2, 6)}

param_grid_range = {'C': np.logspace(-3, 2, 6), 'gamma': logspace(-7, -2, 6)}

for param_grid, ax in zip([param_grid_linear, param_grid_one_log, param_grid_range], axes):
	grid_search = GridSearchCV(SVC(), param_grid, cv=5)
	grid_search.fit(X_train, y_train)
	scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)

	scores_image = mglearn.tools.heatmap(scores, xlabel='gamma', ylabel='C', xticklabels=param_grid['gamma'], yticklabels=param_grid['C'], cmap='viridis', ax=ax)

plt.colorbar(scores_image, ax=axes.tolist())



## search over spaces that are not grids

# a list of dictionaries deals with 'conditional' parameters
param_grid = [{'kernel':['rbf'], 'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}, {'kernel': ['linear'], 'C':[0.001, 0.01, 0.1, 1, 10, 100]}]

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_score_


results = pd.DataFrame(grid_search.cv_results_)
results

### using diffrent cross-validation strategies

### nested cross-validation
# cross_val_score implements cross validation to the whole data set, GridSearchCV implements cross-validation to the training data set
scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), iris.data, iris.target, cv=5)
print('cross-validation scores:', scores)
print('mean cross-validation score:', score.mean())


def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameters_grid):
	# define a function to explain the work flow of nested cross-validation
	outer_scores = []
	# outer_cv is the outer cross-validation split method
	# outer_cv.split(X, y)???
	for training_samples, test_samples in outer_cv.split(X, y):
		# find best parameters using inner cross-validation
		best_parms = {}
		best_score = -np.inf
		# iterate over parameters
		for parameters in parameters_grid:
			cv_scores =[]

			for inner_train, inner_test in inner_cv.split(X[training_samples], y[training_samples]):
				# build classifier given parameters and training data
				clf = Classifier(**parameters)
				clf.fit(X[inner_train], y[inner_train])
				# evaluate on inner test set
				score = clf.score(X[inner_test], y[inner_test])
				cv_scores.append(score)
			# compute mean score over inner folds
			mean_score = np.mean(cv_scores)
			if mean_score > best_score:
				best_score = mean_score
				best_parms = parameters
		build classifier on best parameters using outer training setclf = Classifier(**best_parms)
		clf.fit(X[training_samples], y[training_samples])
		# evaluate the selected parameters using test set
		outer_scores.append(clf.score(X[test_samples], y[test_samples]))
	return np.array(outer_scores)


from sklearn.model_selection import ParameterGrid, StratifiedKFold
scores = nested_cv(iris.data, iris.target, StratifiedKFold(5), StratifiedKFold(5), SVC, ParameterGrid(param_grid))
scores


## Evaluation Metrics Scoring