# This module is for the analysis of the extracted spikes
# For spike sorting, spike train comparison, etc
import config
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import config
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import umap
import numpy as np



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
    plt.savefig("./imgs/GMM Clustering.png")
    plt.show()
    return labels


def umap_dimension_reduction(cutouts):
    fit = umap.UMAP()
    u = fit.fit_transform(cutouts)
    return u


def isi_function(spikes_in_range):
    '''Do ISI (Inter-spike interval) analysis'''
    isis = np.diff(spikes_in_range)  # in seconds
    # Convert to ms
    isis_ms = isis * 1000  

    # Count violations below 1 ms (or 1.5–2 ms if you want to be conservative)
    violations = np.sum(isis_ms < 1.0)
    total_spikes = len(spikes_in_range)
    isis_ms = isis * 1000
    


    print("Total spikes:", total_spikes)
    print("Refractory violations (<1 ms):", violations)
    print("Violation rate (%):", 100 * violations / total_spikes)
    cv_isi = np.std(isis_ms) / np.mean(isis_ms)
    print("CV(ISI):", cv_isi) 
    '''
    CV ≈ 1 → Poisson‑like, irregular
    CV < 0.5 → regular firing
    CV > 1 → bursty / highly irregular
    '''

    # Plot ISI 
    plt.hist(isis_ms, bins=100, range=(0, 50))
    plt.xlabel("ISI (ms)")
    plt.ylabel("Count")
    plt.title("Inter-spike interval histogram")
    plt.savefig("./imgs/ISI Histogram.png")
    plt.show()

