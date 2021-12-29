import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.preprocessing import scale, MaxAbsScaler

cm_bright = ListedColormap(["#FF0000", "#0000FF"])

fig, axes = plt.subplots(3, 2, figsize=(12, 18))

X, y = datasets.make_blobs(n_samples=5000, n_features=2, centers=2, shuffle=True, random_state=3333)
X = scale(X)
transformer = MaxAbsScaler().fit(X)
X = transformer.transform(X)

axes[0, 0].set_xlabel("x[0]")
axes[0, 0].set_ylabel("x[1]")
axes[0, 0].title.set_text("Two Blobs")
axes[0, 0].scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=cm_bright,
    edgecolors="k",
)

X, y = datasets.make_blobs(n_samples=5000, n_features=2, centers=2, shuffle=True, random_state=170, cluster_std=0.7)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)

X = scale(X)
transformer = MaxAbsScaler().fit(X)
X = transformer.transform(X)

axes[0, 1].set_xlabel("x[0]")
axes[0, 1].set_ylabel("x[1]")
axes[0, 1].title.set_text("Two Anisotropically Distributed Blobs")
axes[0, 1].scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=cm_bright,
    edgecolors="k",
)

X, y = datasets.make_circles(n_samples=5000, shuffle=True, noise=0.05, factor=0.5, random_state=3333)
X = scale(X)
transformer = MaxAbsScaler().fit(X)
X = transformer.transform(X)

axes[1, 0].set_xlabel("x[0]")
axes[1, 0].set_ylabel("x[1]")
axes[1, 0].title.set_text("Two Circles")
axes[1, 0].scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=cm_bright,
    edgecolors="k",
)

X, y = datasets.make_blobs(n_samples=5000, cluster_std=[0.5, 2.0], n_features=2, centers=2, shuffle=True,
                           random_state=170)
X = scale(X)
transformer = MaxAbsScaler().fit(X)
X = transformer.transform(X)

axes[1, 1].set_xlabel("x[0]")
axes[1, 1].set_ylabel("x[1]")
axes[1, 1].title.set_text("Two Blobs With Varied Variances")
axes[1, 1].scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=cm_bright,
    edgecolors="k",
)

X, y = datasets.make_blobs(n_samples=10000, n_features=2, centers=2, shuffle=True, random_state=170)
X = np.vstack((X[y == 0][:4500], X[y == 1][:500]))
y = np.concatenate((np.zeros(4500), np.ones(500)))

rand_indices = np.random.permutation(len(X))
X = X[rand_indices]
y = y[rand_indices]

X = scale(X)
transformer = MaxAbsScaler().fit(X)
X = transformer.transform(X)

axes[2, 0].set_xlabel("x[0]")
axes[2, 0].set_ylabel("x[1]")
axes[2, 0].title.set_text("Two Blobs With Varied Number of Samples")
axes[2, 0].scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=cm_bright,
    edgecolors="k",
)

X, y = datasets.make_moons(n_samples=5000, shuffle=True, noise=0.05, random_state=1000)
X = scale(X)
transformer = MaxAbsScaler().fit(X)
X = transformer.transform(X)

axes[2, 1].set_xlabel("x[0]")
axes[2, 1].set_ylabel("x[1]")
axes[2, 1].title.set_text("Two Moons")
axes[2, 1].scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=cm_bright,
    edgecolors="k",
)

plt.savefig(f"clustering_data.png")
