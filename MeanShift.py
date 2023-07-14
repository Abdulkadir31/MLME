from sklearn.cluster import MeanShift
import numpy as np
import matplotlib.pyplot as plt
import json


pcaPath = '.\\data\\size-500\\PCA\\PCA_'
i = 180
# Load the features from the file
filename = pcaPath + 'features_' + str(i) + '.txt'
with open(filename, 'r') as f:
    features = np.array(json.load(f)) 

# Perform Mean Shift clustering
meanshift = MeanShift()
labels = meanshift.fit_predict(features)

# Get unique labels and cluster centers
unique_labels = np.unique(labels)
cluster_centers = meanshift.cluster_centers_

# Visualize the clusters and cluster centers
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed

for label in unique_labels:
    color = colors[label % len(colors)]
    
    # Plot points of the current cluster
    cluster_points = features[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Cluster {label}')

# Plot cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', c='black', label='Cluster Centers')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Mean Shift Clustering')
plt.legend()
plt.show()