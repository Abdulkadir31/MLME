import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import hdbscan

pcaPath = '.\\data\\size-100\\PCA\\PCA_'
i = 60
# Load the features from the file
filename = pcaPath + 'features_' + str(i) + '.txt'
with open(filename, 'r') as f:
    features = np.array(json.load(f)) 

# scaler = MinMaxScaler()
# features = scaler.fit_transform(features)


hdbscan = hdbscan.HDBSCAN(min_samples = 2)
labels = hdbscan.fit_predict(features)
# hdbscan.condensed_tree_.plot(select_clusters=True)
# np.unique(labels)
clusters = labels
print(clusters)    
# # Perform DBSCAN clustering
# dbscan = DBSCAN(eps=5000, min_samples=10)  # Adjust the epsilon and min_samples values as needed
# clusters = dbscan.fit_predict(features)
# print(clusters)

# Visualize the clusters
unique_labels = np.unique(labels)
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
plt.title('HDBSCAN size-100 Clustering')
plt.legend()
plt.savefig('HDBScan(100).png')