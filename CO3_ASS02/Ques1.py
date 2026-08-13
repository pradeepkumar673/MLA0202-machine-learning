import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [19, 21, 20, 23, 31, 22, 35, 23, 64, 30],
    "AnnualIncome": [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
    "SpendingScore": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}
df = pd.DataFrame(data)
print("Raw Data:")
print(df)
features = df[["Age", "AnnualIncome", "SpendingScore"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
wcss = []
k_range = range(1, 6)
for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km.fit(scaled_features)
    wcss.append(km.inertia_)
plt.figure(figsize=(6, 4))
plt.plot(list(k_range), wcss, marker="o")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.close()
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)
df["Cluster"] = clusters
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]
print("\nExplained variance ratio by each principal component:")
print(pca.explained_variance_ratio_)
plt.figure(figsize=(6, 5))
colors = ["red", "green", "blue", "purple", "orange"]
for c in range(optimal_k):
    subset = df[df["Cluster"] == c]
    plt.scatter(subset["PCA1"], subset["PCA2"], c=colors[c], label=f"Cluster {c}")
plt.title("Customer Segments (PCA Reduced to 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("customer_clusters.png")
plt.close()
print("\nFinal Data with Cluster Labels:")
print(df[["CustomerID", "Age", "AnnualIncome", "SpendingScore", "Cluster"]])
print(f"\nOptimal number of customer groups chosen: {optimal_k}")
print("Cluster centers (in scaled feature space):")
print(kmeans.cluster_centers_)