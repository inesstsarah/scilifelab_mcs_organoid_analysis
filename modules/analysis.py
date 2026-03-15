# This module is for the analysis of the extracted spikes
# For spike sorting, spike train comparison, etc
import config
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import config
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import umap



# Choose number of PCA components
PCA_NUMBER = config.PCA_NUMBER


def pca_dimension_reduction(PCA_NUMBER, cutouts):
    scaler = StandardScaler()
    scaled_cutouts = scaler.fit_transform(cutouts)

    pca = PCA()
    pca.fit(scaled_cutouts)
    print(pca.explained_variance_ratio_)
    pca.n_components = PCA_NUMBER
    transformed = pca.fit_transform(scaled_cutouts)
    return transformed


def gmm_clustering(n_components, transformed):
    gmm = GaussianMixture(n_components=n_components, n_init=10)
    labels = gmm.fit_predict(transformed)

    _ = plt.figure(figsize=(8,8))
    for i in range(n_components):
        idx = labels == i
        _ = plt.plot(transformed[idx,0], transformed[idx,1],'.')
        _ = plt.title('Cluster assignments by a GMM')
        _ = plt.xlabel('Principal Component 1')
        _ = plt.ylabel('Principal Component 2')
        _ = plt.axis('tight')
    plt.show()


def umap_dimension_reduction(cutouts):
    fit = umap.UMAP()
    u = fit.fit_transform(cutouts)
    return u

