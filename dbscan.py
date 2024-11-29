import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.vstack((np.random.normal(0, 1, (50, 2)), 
                  np.random.normal(5, 1, (50, 2))))

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=1.0, min_samples=5)
labels = dbscan.fit_predict(data)

# Plot the results
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title("DBSCAN Clustering")
plt.show()
