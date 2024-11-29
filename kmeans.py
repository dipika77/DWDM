import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.vstack((np.random.normal(0, 1, (50, 2)), 
                  np.random.normal(5, 1, (50, 2))))

# Perform k-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the results
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()
