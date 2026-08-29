import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score

digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

print("K-Means Silhouette:", silhouette_score(X_scaled, kmeans_labels))
print("GMM Silhouette:", silhouette_score(X_scaled, gmm_labels))
print("K-Means ARI:", adjusted_rand_score(y, kmeans_labels))
print("GMM ARI:", adjusted_rand_score(y, gmm_labels))

fig, axes = plt.subplots(1, 2, figsize=(12,5))
axes[0].scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, cmap='tab10', s=15)
axes[0].set_title('K-Means')
axes[1].scatter(X_pca[:,0], X_pca[:,1], c=gmm_labels, cmap='tab10', s=15)
axes[1].set_title('GMM')
plt.show()