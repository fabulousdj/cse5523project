from sklearn.cluster import KMeans


def k_means_clustering(img, shape, n_clusters, max_iter, n_init):
    data = img.reshape(shape[0] * shape[1], shape[2])
    clf = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    clf.fit(data)
    return clf.labels_
