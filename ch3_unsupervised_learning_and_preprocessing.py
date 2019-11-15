# unsupervised learning and preprocessing

import numpy as np
import panda as pd
import matplotlib.pyplot as plt
import mglearn

# unsupervised learning, the learning algorithm is just shown the input data and asked to extract knowledge from this data

## types of unsupervised learning
- dimensionality reduction
- clustering, partition data into distinct groups of similar items

## preprocessing and rescaling
mglearn.plots.plot_scaling()
### applying data transformations
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
X_train.shape
X_test.shape

# basic usage of model in sklearn modules, first import the class, second instantiate it, third fit the data, (forth, transform data for data preprocessing)
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

X_test_scaled = scaler.transform(X_test)
# the test set, after scaling, the min and max are not 0 and 1, because the scaler is based on training set
X_test_scaled.min(axis=0)
X_test_scaled.max(axis=0)


### scaling training and test data the same way
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# test_size?
X_train, X_test = train_test_split(X, random_state=5, test_size=.1) 
# plot the training and test data
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
# s=60, marker size?
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label='training set', s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c=mglearn.cm2(1), label='test set', s=60)
axes[0].legend(loc='upper left')
axes[0].set_title('original data')

# scale the data(based on training) using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label='training set', s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label='test set', s=60)
axes[1].set_title('scaled date')

# scale the test set separately, which is bad
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:,1], c=mglearn.cm2(0), label='training set', s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', c=mglearn.cm(1), label='test set', s=60)
axes[2].set_title('scale test set badly')


for ax in axes:
	ax.set_xlabel('feature 0')
	ax.set_ylabel('feature 1')



#### shortcuts and sfficient alernatives for data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit(X).transform(X)
# same result
X_scaled = scaler.fit_transform(X)


### the effect of preprocessing on supervised learning
from sklearn.svm import svc
# cancer datasets
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
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
# svm.coef_?

## Dimensionality reduction, feature extraction, maniford learning

### principle component analysis(PCA)
# an illustraion of PCA
mglearn.plots.plot_pca_illustration()

# apply PCA to cancer datasets
# load data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.data.shape

# relationship between every two features using histogram
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

# ?
ax = ases.ravel()

# 30 features
for i in range(30):
	# cut the data for histogram
	_, bins = np.histogram(cancer.data[:, i], bins=50)
	ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
	ax[i].hist(benign[:, i],bins=bins, color=mglearn.cm3(2), alpha=.5)
	ax[i].set_title(cnacer.feature_name[i])
	# ?
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
# ?
plt.gca().set_aspect('equal')
plt.xlabel('first principle component')
plt.ylabel('second principle component')

# principle components are linear combinations of the original features. the combinations are usually complex, that is not easy to interpret

# the coeffiencts of the original features, combining the PC, are stored in the component_ attribute
pca.component_.shape
pca.component_

# matrix plot the component_
# viridis?
plt.matshow(pca.component_, cmap='viridis')
plt.yticks([0, 1], ['first component', 'second component'])
plt.colorbar()
# ha?
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel('feature')
plt.ylabel('principle components')


#### eigenfaces for feature extraction
# the idea behind feature extraction is that it is possible to find a representation of your data is better suited to analysis than the raw representation 
from sklearn.datasets import fetch_lfe_people
people = fetch_lfe_people(min_faces_per_person=20, resize=0.7)
people
people.images
people.target

image_shape = people.images[0].shape

# subplots_kw?
fix, axes = plt.subplots(2, 5, figsize=(15, 8), subplots_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
	# imshow?
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
	# newline every 3 people? every 2 people?
	if (i + 1) % 3 == 0:
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

scale the grayscale of values to be 0-1 instead of 0-255
X_people = X_people / 255


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

# try data preprocessing, then instantiate PCA
pca = PCA(n_components=100, whitrn=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
X_train_pca.shape
