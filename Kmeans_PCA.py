from sklearn.cluster import KMeans
import numpy as np
import json
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd 

ks = [3, 5, 7, 10, 25, 40, 60, 80, 108]
#ks = [5, 20, 45, 70, 100, 140, 180, 230, 300, 360, 420, 520]
modifiedPath = '.\\data\\size-100\\raster_mod\\'
pcaPath = '.\\data\\size-100\\PCA\\PCA_'
clusterImagePath = '.\\data\\size-100\\ClusterImages\\'
files = os.listdir(modifiedPath)
os.environ['OMP_NUM_THREADS'] = '1'

dataset = []
no_clusters = 16

for file in files:
    image_bw = cv2.imread(modifiedPath+file, cv2.IMREAD_GRAYSCALE)
    dataset.append(image_bw)
dataset = np.array(dataset)
dataset = dataset.reshape(dataset.shape[0],-1)


# Create the directory to save cluster images if it doesn't exist
if not os.path.exists(clusterImagePath):
    os.makedirs(clusterImagePath)

#for i in ks:
i = 60
# Load the features from the file
filename = pcaPath + 'features_' + str(i) + '.txt'
with open(filename, 'r') as f:
    features = np.array(json.load(f)) 

# Perform K-means clustering
kmeans = KMeans(n_clusters=no_clusters, n_init=10)  # Specify the desired number of clusters
clusters = kmeans.fit_predict(features)

print("Cluster labels for Principal components =", i)
print(clusters)

# Save the cluster labels to a file
cluster_labels_filename = pcaPath + 'cluster_labels_' + str(i) + '.txt'
with open(cluster_labels_filename, 'w') as f:
    f.write('\n'.join(map(str, clusters)))
    print("Cluster labels stored in file for Principal components =", i)

# Save cluster images
for cluster_id in range(no_clusters):
    cluster_image_path = clusterImagePath + 'cluster_' + str(cluster_id) + '\\'
    if not os.path.exists(cluster_image_path):
        os.makedirs(cluster_image_path)
    
    for idx, image_idx in enumerate(np.where(clusters == cluster_id)[0]):
        image = dataset[image_idx].reshape(64, 64)
        image_filename = cluster_image_path + 'image_' + str(idx) + '.png'
        cv2.imwrite(image_filename, image)
