# unsupervised learning and preprocessing

# import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

%matplotlib inline

# unsupervised learning, the learning algorithm is just shown the input data and asked to extract knowledge from this data

## types of unsupervised learning
- dimensionality reduction
- clustering, partition data into distinct groups of similar items

## preprocessing and rescaling
# an illustraion of rescaling of data
mglearn.plots.plot_scaling()

### applying data transformations
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

type(cancer)
cancer

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
X_train.shape
X_test.shape


**basic usage of model in sklearn modules**
- step1 import the class
- step2 instantiate it
- step3 fit the data
- step4 transform data for data preprocessing(for data preprocessing models)
- step5 predict(for ml models)

# import model
from sklearn.preprocessing import MinMaxScaler
# instantiate
scaler = MinMaxScaler()
# fit
scaler.fit(X_train)

# transform
X_train_scaled = scaler.transform(X_train)
X_train_scaled.shape
X_train.min(axis=0)
X_train.max(axis=0)
X_train_scaled.min(axis=0)
X_train_scaled.max(axis=0)
# test data most be scaled the same as training data
X_test_scaled = scaler.transform(X_test)
# the test set, after scaling, the min and max are not 0 and 1, because the scaler is based on training set
X_test_scaled.min(axis=0)
X_test_scaled.max(axis=0)


### scaling training and test data the same way
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
make_blobs?

X
_
X.shape

# test_size, float between 0 and 1, the proportion of the dataset generating test set; int, the number of dataset generating test set
X_train, X_test = train_test_split(X, random_state=5, test_size=.1) 
X_test.shape

# plot the training and test data
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# subplot 1
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label='training set', s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c=mglearn.cm2(1), label='test set', s=60)
axes[0].legend(loc='upper left')
axes[0].set_title('original data')

# subplot 2
# scale the data(based on training) using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label='training set', s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label='test set', s=60)
axes[1].set_title('scaled date')

# subplot 3
# scale the test set separately, which is bad
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:,1], c=mglearn.cm2(0), label='training set', s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', c=mglearn.cm2(1), label='test set', s=60)
axes[2].set_title('scale test set badly')

for ax in axes:
	ax.set_xlabel('feature 0')
	ax.set_ylabel('feature 1')



#### shortcuts and efficient alernatives for data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit(X).transform(X)
X_scaled[:10]
# same result
X_scaled = scaler.fit_transform(X)
X_scaled[:10]


### the effect of preprocessing on supervised learning
from sklearn.svm import SVC
# cancer datasets
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# C trade-off parameter, small C makes the model more gerenal
svm = SVC(C=100)
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

# preprocessing using MinMaxScaler(0-1 scaling)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
svm.score(X_test_scaled, y_test)


# preprocessing using StandardScaler(0 mean, 1 variance)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
svm.score(X_test_scaled, y_test)

svm.
# svm.<Tab>
# show svm methods


## Dimensionality reduction, feature extraction, maniford learning

### principle component analysis(PCA)
# an illustraion of PCA
mglearn.plots.plot_pca_illustration()

# apply PCA to cancer datasets
# load data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()

cancer.data.shape
cancer.target.shape
cancer.data[cancer.target == 0].shape

# relationship between benign and malignant on every feature using histogram
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

# Return a flattened array
ax = axes.ravel()

# 30 features
for i in range(30):
	# cut the data for histogram
	_, bins = np.histogram(cancer.data[:, i], bins=50)
	ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
	ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
	ax[i].set_title(cancer.feature_names[i])
    # y轴标签为空
	ax[i].set_yticks(())
ax[0].set_xlabel('feature magnitude')
ax[0].set_ylabel('frequency')
ax[0].legend(['magnitude', 'benign'], loc='best')
# make the layout suit for the figsize
fig.tight_layout()


# scale the cancer data using 0 mean, 1 variance scaler(StandardScaler)
scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)


# PCA rotate and shift the data, keeps all principle components, then specify how many components to be left for reducing the dimensionality
from sklearn.decomposition import PCA
# keep the first two components
pca = PCA(n_components=2)
pca.fit(X_scaled)
# like the data preprocessing, dimensionality reduction procedure need the transform step
X_pca = pca.transform(X_scaled)
print('original shape: {}'.format(str(X_scaled.shape)))
print('reduced shape: {}'.format(str(X_pca.shape)))


plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc='best')
# plt.gca, get the current axes
# axes.set_aspect, set the aspect of the axis scaling, i.e. the ratio of y-unit to x-unit
plt.gca().set_aspect('equal')
plt.xlabel('first principle component')
plt.ylabel('second principle component')


# principle components are linear combinations of the original features. the combinations are usually complex, that is not easy to interpret

# the coeffiencts of the original features, combining the PC, are stored in the components_ attribute
pca.components_.shape
pca.components_

# X_pca and X_pca1 are equal
X_pca
X_pca1 = np.dot(X_scaled, pca.components_.T)
X_pca1


# matrix plot of the components_
# viridis, a form of colormap, perceptually uniform shades of blue-green-yellow
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ['first component', 'second component'])
plt.colorbar()
# ha, horizontalalignment, {'center', 'right', 'left'}
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='center')
plt.xlabel('feature')
plt.ylabel('principle components')


#### eigenfaces for feature extraction
# the idea behind feature extraction is that it is possible to find a representation of your data that is better suited to analysis than the raw representation 
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

people.keys()

people.images
people.target
type(people.target)
people.target_names

# get the shape of every image. it will be used for components data reshape
image_shape = people.images[0].shape
image_shape

# subplot_kw, key-word parameter, xticks, yticks none
fix, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
# zip will cut the elements with one with the lowest length
for target, image, ax in zip(people.target, people.images, axes.ravel()):
	ax.imshow(image)
	ax.set_title(people.target_names[target])


# 3023 images 87*65 pixels
people.images.shape
# 62 different people
len(people.target_names)

# count how often each target appears
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
	print('{0:25} {1:3}'.format(name, count), end='    ')
	if (i + 1) % 3 == 0:
		# newline
		print()


people.target.shape
mask = np.zeros(people.target.shape. dtype=np.bool)
# select 50 images of each person
# ?
# people.target
# np.where(people.target == people.target[1])
for target in np.unique(people.target):
	mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grayscale of values to be 0-1 instead of 0-255
X_people = X_people / 255
X_people.shape

from sklearn.neighbors import KNeighborsClassifier
# stratify?
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
X_train.shape

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

# computing distances in the original pixel space is a bad way to measure similatity between faces. using the original representation, compare the grayscale value of ench individual pixel to the value of the pixel at the same pisition in the other image

# whitening option of PCA, rescales the principle components to have the same scale. rotate and rescale the data as a circle 
mglearn.plots.plot_pca_whitening()

# try data preprocessing, then instantiate PCA instead of using parameter whiten
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
# principle components are representation such as  alignment of face(like position of eyes), lighting of the image, etc.
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
X_train_pca.shape


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
knn.score(X_test_pca, y_test)

# how to interpret
pca.components_.shape

# return of plt.subplots?
fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
	ax.imshow(component.reshap	(image_shape), cmap='viridis')
	ax.set_title('{}. component'.format((i + 1)))


# PCA, rotate the original data to new axes, keep the components with high variance and drop the low variance

# an illustraion of PCA, reconstruct the original data using some components
mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

# scatter plot of the faces using the first two principle components labeled by the people the point satands for
mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel('first principle component')
plt.ylabel('second principle component')


### Non-Negative Matrix Factorization(NMF)
# NMF, the components and coeffiencts are non-negative. it can only be applied to data where each feature is non-negative such as an audio track of multiple people skeaking, or music with many instruments. since the components are non-negative, they are interpretable than PCA components which may be negative

# a two dimensional NMF
mglearn.plots.plot_nmf_illustration()


# all the points in the data can be written as a positive combination of the NMF components. if we use only one component in NMF, it will be the one point tp the mean which best explain the data. all components in NMF play an equal part in contrast with PCA components

#### apply NMF to face images
mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)


# NMF is not used for reconstruct or encode data, but for finding intersting patterns within the data
# examples for nmf components, such as faces turn right, face highlight in the head, etc.
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
# X_train from fetch_lfw_people dataset
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks':(), 'yticks':()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
	# reshape the nmf component as the people_image
	ax.imshow(component.reshape(image_shape))
	ax.set_title('{}. component'.format(i))



compn = 3
# sort by 3rd component, plot first 10 images
# np.argsort?
# X_train_nmf[:, 3]?
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
	ax.imshow(X_train[ind].reshape(image_shape))

compn = 7
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
	ax.imshow(X_train[ind].reshape(image_shape))


S = mglearn.datasets.make_signals()
S
S.shape
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel('time')
plt.ylabel('signal')


# 0-1 uniform dist.
A = np.random.RandomState(0).uniform(size=(100, 3))
# mix data into a 100-dimensinal state
X = np.dot(S, A.T)
X.shape

# use NMF to recover the three signals
nmf = NMF(n_components=3, random_state=42)
# nmf transformed dataset
S_ = nmf.fit_transform(X)
S_.shape

# use PCA
pca = PCA(n_components=3)
# pca transformed data
H = pca.fit_transform(X)

models = [X, S, S_, H]
names = ['observations(first three measurements', 'actual signals', 'NMF recovered signals', 'PCA recovered signals']

# gridspec_kw?
fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hsapce': .5}, subplot_kw={'xticks':(), 'yticks':()})


for model, name, ax in zip(models, names, axes):
	ax.set_title(name)
	# plot original data and transformed data
	ax.plot(model[:, :3], '-')


### other decomposition methods
- ICA independent component analysis
- FA factor analysis
- sparse coding(dictionary learning)
(decomposition methods webpage)[https://scikit-learn.org/stable/modules/decomposition.html]

### Manifold learning with t_SNE（流行学习）

# maniford learning algorithms are mainly aimed at visualization. these method only provide a new representation of the original data(transform training data to new features), cannot transform new data(test data). 
# it can explore the existence data, cannot used in supervised learning for prediction
# t-SNE(t-distributed stochastic neighbor embedding), find a two dimensional representation of the original data to preserve the distances between points best. it tries to preserve the information indicating which points are neighbors to each others


from sklearn.datasets import load_digits
# handwritten digit between 0 and 9
digits = load_digits()
digits
digits.data
digits.target
digits.data.shape

fix, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks':(), 'yticks':()})
for ax, img in zip(axes.ravel(), digits.images):
	ax.imshow(img)



# biuld a PCA model with 2 components
from sklearn.decomposition import PCA 
pca = PCA(n_components=2)
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)
colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525', '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
plt.figure(figsize=(10, 10))
# first component's scale
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max)
# second component's scale
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
	# plot the digits as text
	# digits_pca[i, 0]?
	# digits.target[i]?
	# fontdict?
	plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), colors = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9})
plt.xlabel('first principle component')
plt.ylabel('second principle component')




from sklearn.maniford import TSNE
# random_state?
tsne = TSNE(random_state=42)
# fit and transform chain
digits_tsne = tsne.fit_transform(digits.data)


plt.figure(figsize=(10, 10))
# first component's scale
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max)
# second component's scale
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())
for i in range(len(digits.data)):
	# plot the digits as text
	# digits_pca[i, 0]?
	# digits.target[i]?
	# fontdict?
	plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), colors = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9})
plt.xlabel('t-SNE feature 0')
plt.ylabel('t-SNE feature 1')



### Clustering
#### k-Means Clustering
- step 0: assigning n points randomly as the original centers
- step 1: assigning each data points to the closest cluster center
- step 2: setting each cluster center as the mean of the data points that are assigned to it
- step 3: repeat step1 and step2 until the assignment no longer changes

# an illustraion of k-Means cluster
mglearn.plots.plot_kmeans_algorithm()

mglearn.plots.plot_kmeans_boundaries()


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(random_state=1)

# n_clusters default is 8
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
# each data point in X is assigned a cluster label
kmeans.labels_
# assign cluster labels to new points using predict method, for the training data set, the prediction is the same as labels_
kmeans.predict(X)



mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
# cluster_centers_ stores cluster center
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)


# compare 2 clusters with 5 clusters
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])


#### failure cases og k-menas cluster
# make_blobs?
X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)

mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
plt.legend(['cluster 0', 'cluster 1', 'cluster 2'], loc='best')
plt.xlabel('feature 0')
plt.ylabel('feature 1')



X, y = make_blobs(random_state=170, n_samples=600)
X


rng = np.random.RandomState(74)
transformation = rng.normal(size=(2, 2))
transformation
# linear transformation
X = np.dot(X, transformation)
X

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3)
# c?
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', c=[0, 1, 2], s=100, linewidth=2, cmap=mglearn.cm3)
plt.xlabel('feature 0')
plt.ylabel('feature 1')


# two_moons data
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, moise=0.05, random_state=0)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', c=[mglearn.cm2(0), mglearn.cm2(1)], s=100. linewidth=2)
plt.xlabel('feature 0')
plt.ylabel('feature 1')


# seeing k-means as a decomposition method, where each point is represented using a single component(the cluster center), is called vector quantization


# using pca, nmf, kmeans train people data
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)
kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)

fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle('extracted components')
for ax, comp_kmeans,  comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
	# reshape the components_ (cluster_centers_) as a image shape, then display the image
	# ? n=100, why components_ imshow only 5
	ax[0].imshow(comp_kmeans.reshape(image_shape))
	ax[1].imshow(comp_pca.reshape(image_shape), cmap='virdis')
	ax[2].imshow(comp_nmf.reshape(image_shape))

axes[0, 0].set_ylabel('kmeans')
axes[1, 0].set_ylabel('pca')
axes[2, 0].set_ylabel('nmf')



# reconstruct the test data using different models
# ?
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)
# ?
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test_scaled))
# ?
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]

fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(8, 8))
fig.suptitle('reconstruction')
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca, X_reconstructed_nmf):
	ax[0].imshow(orig.reshape(image_shape))
	ax[1].imshow(rec_kmeans.reshape(image_shape))
	ax[2].imshow(rec_pca.reshape(image_shape))
	ax[3].imshow(rec_nmf.reshape(image_shape))

axes[0, 0].set_ylabel('original')
axes[1, 0].set_ylabel('kmeans')
axes[2, 0].set_ylabel('pca')
axes[3, 0].set_yticks('nmf')




# for 2-dimensional data, using pca or nmf to reduce dimension is not a good idea, but kmeans with large cluster number may give a expressive representation
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.labels_
y_pred = kmeans.predict(X)

# plot scatter of X
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
# plot scatter of cluster center
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60, marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
plt.xlabel('feature 0')
plt.ylabel('feature 1')


# n_clusters=10 makes the 2-dimensional original data with 10 features, which is the 10 distance_features to 10 different cluster center
distance_features = kmeans.transform(X)
distance_features.shape
distance_features


# with 10 feature, apply linear classification model?
# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()

# apply kmeans cluster to large dataset using MiniBatchKMeans class


### Agglomerative Clustering (凝聚聚类)
step1: each point is a single cluster
step2: merge similar clusters with some specific linkage criteria
step3: repeat step2 until some stopping criterion is reached

- linkage
	- ward, the default choice, it picks two clusters to merge such that the variance within all clusters increases the least
	- average, it merges the two clusters that have the smallest average distance betweent all th
	- complete, it merges the two clusters that hace the smallest maximum distance between their points

- stopping criterion
	- number of clusters

# an illustraion of agglomerative clustering
mglearn.plots.plot_agglomerative_algorithm()



# agglomerative clustering has no predict method for new data points because of the way it works, but has the chain method fit_predict working on training dataset
# agglomerative clustering requires user to specify the number of clusters
from sklearn.cluster import AgglomerativeClustering
X, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.xlabel('feature 0')
plt.ylabel('feature 1')

# an illustraion of agglomerative clustering with iterative procedure
mglearn.plots.plot_agglomerative()



# dendrogram, a visualization of hierarchical clustering
# SciPy provides dendrogram


from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)
# apply the ward clustering to array X
# SciPy ward function returns an array that specifies the distances bridged when performing agglomerative clustering
linkage_array = ward(X)
# plot dendrogram for the linkage_array containing the distances between clusters
dendrogram(linkage_array)

ax = plt.gca()
# ?
bounds = ax.get_xbound()
# plot the bound line
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
# bounds[1]?
ax.text(bounds[1], 7.25, 'two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, 'three clusters', va='center', fontdict={'size': 15})
plt.xlabel('sample index')
plt.ylabel('cluster distance')



### DBSCAN
# DBSCAN stands for density-based spatial clustering of applications with noise.
# it does not require to set the number of clusters priorily
# it can capture clusters of complex shapes such as half moon shape
# it can identify points that are not part of any cluster

# DBSCAN's clusters form dense regions of data, separated by regions that are relatively empty
# points within a dense region are called core samples(core points). if there are at least min_samples(parameter of DBSCAN) points within a distance of eps(parameter of DBSCAN) to a given data point, that point is classified as a core sample


step1: DBSCAN picks an arbitrary point to start, say point A
step2: find all points within distance eps of A, if the number of points is less than min_samples, A is labeled as noise, else A will be assigned a cluster label
step3: all neighbors within eps of A are visited, if it has not been labeled, it willed labeled the same as A, if it is a core points, its neighbors are visited in turn
step4: clusters grows until no more points left


# build a DBSCAN for a sythetic dataset
# DBSCAN does not allow predictions on new test data
from sklearn.cluster import DBSCAN
X, y = make_blobs(random_state=0, n_samples=12)
X
y

# use default parameter in DBSCAN
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
# labele -1 indicates all points assigned as noise
clusters


# scatter plot
mglearn.discrete_scatter(X[:, 0], X[:, 1], clusters)

# clusters assignments for different parameter values of min_samples and eps
mglearn.plots.plot_dbsacn()



# after data scaling using StandardScaler or MinMaxScaler may help to find a good setting for eps
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# rescale the training data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_transform(X_scaled)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel('feature 0')
plt.ylabel('feature 1')



### caompare and evaluate clustering algorithms
# ARI(adjusted rand index) and NMI(normalized mutual information) are measurements to assess the outcome of a clustering algorithms relative to a ground truth clustering

from sklearn.metrics.cluster import adjusted_rand_score
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# rescale the training data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})

# a list of clusters algorithms
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

# create a random cluster assignment for reference plot in axes[0]
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
random_clusters

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
axes[0].set_title('random assignment - ARI: {:.2f}'.format(adjusted_rand_score(y, random_clusters)))

for ax, algorithm in zip(axes[1:], algorithms):
	# fit the cluster algorithm
	clusters = algorithm.fit_predict(X_scaled)
	# scatter X_scaled with clusters predict from different algorithm
	ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
	# set title with ARI
	ax.set_title('{} - ARI: {:.2f}'.format(algorithm.__class__.__name__, adjusted_rand_score(y, clusters)))


# compare accurary_score with ARI
from sklearn.metrics import accurary_score

clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]

# accurary_score requires the assigned clusters labels to exactly match the ground truth
# ARI only cares whether the points are in the same cluster
accurary_score(clusters1, clusters2)
adjusted_rand_score(clusters1, clusters2)


#### other metrics evaluating the clustering
# the silhouette score computes the compactness of a cluster, high score is better, range from 0 to 1

from sklearn.metrics.cluster import silhouette_score

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# rescale X to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# subplot_kw: Dict with keywords passed to the add_subplot call used to create each subplot
fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks':(), 'yticks':()})

# generate random cluster
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
# plot random cluster in axes[0]
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)

algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

for ax, algorithm in zip(axes[1:], algorithms):
	clusters = algorithm.fit_predict(X_scaled)
	# plot clusters predicted from different algorithm
	ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
	# set title with silhouette score
	ax.set_title('{}: {:.2f}'.format(algorithm.__class__.__name__, silhouette_score(X_scaled, clusters))



#### DBSCAN on faces datasets












