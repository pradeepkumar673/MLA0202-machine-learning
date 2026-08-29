import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.mixture import GaussianMixture
data = {
    "Sample": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Alcohol": [14.23, 13.20, 13.16, 14.37, 13.24, 14.20, 14.39, 14.06, 14.83, 13.86],
    "MalicAcid": [1.71, 1.78, 2.36, 1.95, 2.59, 1.76, 1.87, 2.15, 1.64, 1.35],
    "Ash": [2.43, 2.14, 2.67, 2.50, 2.87, 2.45, 2.45, 2.61, 2.17, 2.27],
    "Alcalinity": [15.6, 11.2, 18.6, 16.8, 21.0, 15.2, 14.6, 17.6, 14.0, 16.0],
    "Magnesium": [127, 100, 101, 113, 118, 112, 96, 121, 97, 98],
    "Phenols": [2.80, 2.65, 2.80, 3.85, 2.80, 3.27, 2.50, 2.60, 2.80, 2.98]
}
df = pd.DataFrame(data)
print("Raw Wine Data:")
print(df)
X = df.drop(columns=["Sample"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
fa = FactorAnalysis(n_components=2, random_state=42)
fa_result = fa.fit_transform(X_scaled)
ica = FastICA(n_components=2, random_state=42, max_iter=1000)
ica_result = ica.fit_transform(X_scaled)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
df["GMM_Cluster"] = gmm_labels
print("\nGaussian Mixture Model cluster assignments:")
print(df[["Sample", "GMM_Cluster"]])
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["PCA", "Factor Analysis", "ICA"]
results = [pca_result, fa_result, ica_result]
colors = ["red", "blue"]
for ax, title, result in zip(axes, titles, results):
    for cluster in np.unique(gmm_labels):
        mask = gmm_labels == cluster
        ax.scatter(result[mask, 0], result[mask, 1], c=colors[cluster], label=f"Cluster {cluster}")
    ax.set_title(f"{title} + GMM Clusters")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
plt.tight_layout()
plt.savefig("wine_dimensionality_comparison.png")
plt.close()
print("\nComparison of first two transformed values per method (Sample 1):")
print(f"PCA: {pca_result[0]}")
print(f"Factor Analysis: {fa_result[0]}")
print(f"ICA: {ica_result[0]}")
print("\nGMM converged:", gmm.converged_)
print("GMM means (in scaled feature space):")
print(gmm.means_)