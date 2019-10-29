
# chap1 introduction

from scipy import sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


eye = np.eye(4)
print("numpy array:\n%s" % eye)

sparse_matrix = sparse.csr_matrix(eye)
print("\nScipy sparse CSR matrix:\n%s" % sparse_matrix)

import sklearn

# version of a package
print("numpy version: %s" % np.__version__)
print("scikit-learn version: %s" % sklearn.__version__)



# first ml example
from sklearn.datasets import load_iris
iris = load_iris()

iris.key()
print(iris['DESCR'][:193] + "\n...")
iris['target_names']
type(iris['data'])
iris['data'].shape #150*4
iris['data'][:5]
type(iris['target'])
iris['target'].shape
iris['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
X_train.shape
X_test.shape

fig, ax = plt.subplot(3, 3, figsize-(15, 15))
plt.suptitle('iris_pairplot')

for i in range(3):
	for j in range(3):
		ax[i, j].scatter(X_train[:, j], X_train[:, i+1], c=y_train, s=60)
		ax[i, j].set_xticks(())
		ax[i, j].set_yticks(())
		if i == 2:
			ax[i, j].set_xlabel(iris['feature_names'][j])
		if j == 0:
			ax[i, j].set_ylabel(iris['feature_names'][i+1])
		if j > i:
			ax[i, j].set_visible(False)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')

X_new = np.array([5, 2.9, 1, 0.2])
prediction = knn.predict(X_new)
prediction

iris['target_names'][prediction]

# evaluate the model

y_pred = knn.predict(X_test)
np.mean(y-pred == y_test)

knn.score(X_test, y_test)



# chap2 supervised learning


## classification and regression
import mglearn
X, y = mglearn.datasets.make_forge()
plt.scatter(X[:,0], X[:,1], c=y, s=60, cmap=mglearn.cm2)
print('X.shape: %s' % (X.shape,))

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, '0')
plt.plot(X, -3*np.ones(len(X)), 'o')
plt.ylim(-3.1, 3.1)


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()

print(cancer.data.shape)
print(cancer.target_names)
np.bincount(cancer.target)

cancer.feature


from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)

# derived features from the feature of boston
X, y = mglearn.datasets.load_extended_boston()
print(X.sha)


### k_neighbors classification
mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.title('forge_one_neighbor')

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.title('forge_three_neighbor')

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

clf.predict(X_test)
clf.score(X_test, y_test)



fig, axes = plt.subplot(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
	clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
	mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
	ax.set_title('{} neighbor(s)'.format(n_neighbors))
	ax.set_xlabel('feature 0')
	ax.set_ylabel('feature 1')
axes[0].legend(loc=3)



from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target stratify=cancer.target, random_state=66)

training_accuracy =[]
test_accuracy = []

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
	clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	clf.fit(X_train, y_train)
	training_accuracy.append(clf.score(X_train, y_train))
	test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label='training accuracy')
plt.plot(neighbors_settings, test_accuracy, label='test accuracy')
plt.ylable('accuracy')
plt.xlabel('n_neighbors')
plt.legend()


### k-neighbors regression
mglearn.plots.plot_knn_regression(n_neighbors=1)
mglearn.plots.plot_knn_regression(n_neighbors=3)


from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
reg.predict(X_test)
# knn regression score returns the $R^2$, $R^2 = corr(y, x)^2$ 衡量线性相关程度，1-回归误差的波动/y的波动=x的波动以及xy存在的线性关系在y的波动中的占比
reg.score(X_test, y_test)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
	reg = KNeighborsRegressor(n_neighbors=n_neighbors)
	reg.fit(X_test, y_train)
	ax.plot(line, reg.predict(line))
	ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
	ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
	ax.set_title('{} neighbor(s)\n train score:{:.2f} test score:{:.2f}'.format(n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)))
	ax.set_xlabel('feature')
	ax.set_ylabel('target')
axes[0].legend(['modle predictions', 'training data/target', 'test data/target'], loc='best')








### linear models
mglearn.plots.plot_linear_regression_wave()

from sklearn.linear_model import LinearRegession

X, y = mglearn.datasets.make_wave(n_samples=60)
# 数据处理流程，1.导入数据，2.绘图
plt.plot(X, y, 'o')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegession().fit(X_train, y_train)
lr.coef_
lr.intercept_
# score for OLS is R^2
lr.score(X_train, y_train)
lr.score(X_test, y_test)


X, y = mglearn.datasets.load_enxtened_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegession().fir(X_train, y_train)
lr.coef_
lr.intercept_
lr.score(X_train, y_train)
lr.score(X_test, y_test)


### Ridge regression
from sklearn.linear_model import Ridge
# using ridge regression, the coefficients of the regression function are close to 0, which make the coefficient index-coefficient magnitude line look straight, just like a ridge
ridge = Ridge().fit(X_train, y_train)
ridge.score(X_train, y_train)
ridge.score(X_test, y_test)

# # alpha parameters, trade-off between performance on the training set and generalization, alpha high, model restrict, coef_ close to 0
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge10.score(X_train, y_train)
ridge10.score(X_test, y_test)

ridge01 = Ridge(alpha=.1).fit(X_train, y_train)
ridge01.score(X_train, y_train)
ridge01.score(X_test, y_test)

plt.plot(ridge.coef_,'s', label='alpha=1')
plt.plot(ridge10.coef_,'^', label='alpha=10')
plt.plot(ridge01.coef_,'v',label='alpha=0.1')
plt.plot(lr.coef_, 'o', label='LinearRegession')
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')
plt.ylim(-25, 25)
plt.legend()


mglearn.plots.plot_ridge_n_samples()


### Lasso
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
lasso.score(X_train, y_train)
lasso.score(X_test, y_test)
# number of features used
np.sum(lasso.coef_ != 0)

# alpha controls how strongly coefficients are pushed towards 0, the smaller alpha is, more complex the model is. ?max_iter(the maximum number of iterations to run)
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
lasso001.score(X_train, y_train)
lasso001.score(X_test, y_test)
np.sum(lasso001.coef_ != 0)


lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
lasso00001.score(X_train, y_train)
lasso00001.score(X_test, y_test)
np.sum(lasso00001.coef_ != 0)


plt.plot(lasso.coef_, 's', label='alpha=1')
plt.plot(lasso001.coef_, '^', label='alpha=0.01')
plt.plot(lasso00001.coef_, 'v', label='alpha=0.0001')
plt.plot(ridge01.coef_, 'o', label='ridge alpha=0.1')
plt.legend(ncol=2, loc(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')



## linear model for classification

### logistic regression & support vector machine
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for  model, ax in zip([LinearSVC(), LogisticRegression()], axes):
	clf = model.fit(X, y)
	mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
	mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
	ax.set_title('{}'.format(clf.__class__.__name__))
	ax.set_xlabel('feature 0')
	ax.set_ylabel('feature 1')
axes[0].legend()

# higher C values correspond to less regularization
mglearn.plots.plots_linear_svs_regularization()


### LinearRegession
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
logreg.score(X_train, y_train)
logreg.score(X_test, y_test)

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
logreg100.score(X_train, y_train)
logreg100.score(X_test, y_test)


logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
logreg001.score(X_train, y_train)
logreg001.score(X_test, y_test)

plt.plot(logreg.coef_.T, 'o', label='C=1')
plt.plot(logreg100.coef_.T, '^', label='C=100')
plt.plot(logreg001.coef_.T, 'v', label='C=0.01')
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')
plt.legend()


for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
	lr_l1 = LogisticRegression(C=C, penalty='l1').fit(X_train, y_train)
	lr_l1.score(X_train, y_train)
	lr_l1.score(X_test, y_test)
	plt.plot(lr_l1.coef_.T, marker, label='C={:.3f}'.format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')
plt.ylim(-5, 5)
plt.legend(loc=3)



### Linear models for multiclass classification
#### support vector machine
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel('feature 0')
plt.ylabel('feature 1')plt.legend(['class 0', 'class 1', 'class 2'])


linear_svm = LinearSVC().fit(X, y)
linear_svm.coef_.shape
linear_svm.intercept_.shape

mglearn.discrete_scatter(X[:, 0], X[:,1], y)
line = np.linspace(-15, 15)
for coef, intercepte, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
	plt.plot(line, -(line * coef[0] + intercepte) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel('feature 0')
plt.ylabel('feature 1')
plt.legend(['class 0', 'class 1', 'class 2', 'line 0', 'line 1', 'line2'], loc=(1.01, 0.3))


mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
line = np.linspace(-15, 15)
for coef, intercepte, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
	plt.plot(line, -(line * coef[0] + intercepte) / coef[1], c=color)
plt.legend(['class 0', 'class 1', 'class 2', 'line 0', 'line 1', 'line 2'], loc(1.01, 0.3))
plt.xlabel('feature 0')
plt.ylabel('feature 1')
linear_svm.predict()?


### naive bayes

# The BernoulliNB classifier counts how often every feature of each class is not zero. for example
X = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1], [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
counts = {}
for label in np.unique(y):
	# iterate ocer each class
	# count (sum) entries of 1 per feature
	counts[label] = X[y == label].sum(axis=0)
print('feature counts:\n{}'.format(counts))
#find examples in official documentation of naive bayes


### Decision trees
mglearn.plots.plot_animal_tree()

from sklearn.tree import DecisionTreeClassifier
cancer = load_breast_cancer
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
tree.score(X_train, y_train)
tree.score(X_test, y_test)


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
tree.score(X_train, y_train)
tree.score(X_test, y_test)

# visualize decision trees
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names, impurity=False, filled=True)

import graphviz

with open('tree.dot') as f:
	dot_graph = f.read()
graphviz.Source(dot_graph)

# feature importances
tree.feature_importances_

# visualize feature importances
def plot_feature_importances_cancer(model):
	n_features = cancer.data.shape[1]
	plt.barh(range(n_features), model.feature_importances_, align='center')
	plt.yticks(np.arange(n_features), cancer.feature_names)
	plt.xlabel('feature importance')
	plt.ylabel('feature')

plot_feature_importances_cancer(tree)


tree = mglearn.plots.plot_tree_not_monotone()
display(tree)

# DecisionTreeRegressor vs linearRegressor
import pandas as pd
ram_prices = pd.read_csv('data/ram_price.csv')

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel('year')
plt.ylabel('price in $/Mbyte')


from sklearn.tree import DecisionTreeRegressor
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

X_train = data_train.date[:, np.newaxis]
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegession().fit(X_train, y_train)

X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = npexp(pred_tree)
price_lr = np.exp(pred_lr)


plt.semilogy(data_train.date, data_train.prcie, label='train data')
plt.semilogy(data_test.date, data_test.prcie, label='test data')
plt.semilogy(ram_prices.date, price_tree, label='tree prediction')
plt.semilogy(ram_prices.date, price_lr, label='linear prediction')
plt.legend()


### Ensembles of Decision Trees
#### Random forests
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test
 = train_test_split(X, y, stratify=y, random_state=42)

 forest = RandomForestClassifier(n_estimators=5, random_state=2)
 forest.fit(X_train, y_train)

 fig, axes = plt.subplots(2, 3, figsize=(20, 10))
 for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
 	ax.set_title('trss {}'.format(i))
 	mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

 mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
 axes[-1, -1].set_title('random forset')
 mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)



#### Gredient boosting forests

from sklearn.ensemble import GredientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

gbrt = GredientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
gbrt.score(X_train, y_train)
gbrt.score(X_test, y_test)

# limit the depth of tree to generalize the model
gbrt = GredientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
gbrt.score(X_train, y_train)
gbrt.score(X_test, y_test)


# lower the learning rate to generalize the model
gbrt = GredientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
gbrt.score(X_train, y_train)
gbrt.score(X_test, y_test)

# visulize the feature importances
gbrt = GredientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
plot_feature_importances_cancer(gbrt)


#### Kernelized support vector machines

X, y = make_blobs(centers=4, random_state=8)
# to make it as class
y = y % 2

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('feature 0')
plt.ylabel('feature 1')

# use svm to classify
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)

# not good
mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('feature 0')
plt.ylabel('feature 1')

# add the squared first feature
X_new = np.hstack([X, X[:, 1:] ** 2])
from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)

mask = y == 0
# calss 1
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
# class 2
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
ax.set_xlabel('feature 0')
ax.set_ylabel('feature 1')
ax.set_zlabel('feature 1 ** 2')

linear_svm_3d = LinearSVC().fit(X_new, y)

coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] ** YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
ax.set_xlabel('feature 0')
ax.set_ylabel('feature 1')
ax.set_zlabel('feature 0 ** 2')

# 投影到xy平面，得到决策边界
ZZ = YY ** 2
dec = linear_svm_3d.decision_funtion(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('feature 0')
plt.ylabel('feature 1')


from sklearn.svm import SVC
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
















