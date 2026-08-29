import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

wine = load_wine()
X = wine.data
y = wine.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)

ica = FastICA(n_components=2, random_state=42, max_iter=1000)
X_ica = ica.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18,5))
for ax, data, title in zip(axes, [X_pca, X_fa, X_ica], ['PCA', 'Factor Analysis', 'ICA']):
    ax.scatter(data[:,0], data[:,1], c=y, cmap='viridis', s=40, edgecolor='k')
    ax.set_title(title)

plt.show()

print(pca.explained_variance_ratio_)
print(pca.components_)
print(fa.components_)
print(ica.mixing_)