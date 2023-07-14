from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import DBSCAN

pcaPath = '.\\data\\size-500\\PCA\\PCA_'
i = 180
# Load the features from the file
filename = pcaPath + 'features_' + str(i) + '.txt'
with open(filename, 'r') as f:
    features = np.array(json.load(f)) 
    
# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=25)  # Adjust the epsilon and min_samples values as needed
clusters = dbscan.fit_predict(features)
print(clusters)

# Visualize the clusters
unique_labels = np.unique(clusters)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed

for label in unique_labels:
    if label == -1:
        # Noise points are plotted in black
        color = 'k'
    else:
        color = colors[label % len(colors)]
    
    # Plot points of the current cluster
    cluster_points = features[clusters == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f'Cluster {label}')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.legend()
plt.show()