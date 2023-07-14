from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import warnings

def warn(*args, **kwargs):
    pass


warnings.warn = warn
os.environ['OMP_NUM_THREADS'] = '1'

pcaPath = '.\\data\\size-500\\PCA\\PCA_'
# ks = [3, 5, 7, 10, 25, 50, 60, 80, 108]
ks = [3, 10, 50, 100, 170, 200, 300, 450, 519]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'aqua', 'teal']
count=0
for i in ks:
    # i = 170
    # Load the features from the file
    features = []
    filename = pcaPath + 'features_' + str(i) + '.txt'
    # print(i)
    with open(filename, 'r') as f:
        features = np.array(json.load(f))

    # Perform K-means clustering for different values of k
    k_values = range(2, 25)  # Range of k values to try
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
    label = str(i)+' Features'
    
    # Plot the elbow curve
    plt.plot(k_values, inertias, 'bo-',label=label, c=colors[count])
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    count += 1
plt.legend()
plt.title('Elbow Method for Size-500 ')
plt.savefig('./Elbow/size-100/Elbow Method(500).png' )
