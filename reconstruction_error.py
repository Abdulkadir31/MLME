import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import mean_squared_error
import json
import os 

modifiedPath = '.\\data\\size-100\\raster_mod\\'
pcaPath = '.\\data\\size-100\\PCA\\PCA_'
files = os.listdir(modifiedPath)
ks = [3, 5, 7, 10, 25, 40, 60, 80, 108]

dataset = []
for file in files:
    image_bw = cv2.imread(modifiedPath + file, cv2.IMREAD_GRAYSCALE)
    dataset.append(image_bw)
dataset = np.array(dataset)
dataset = dataset.reshape(dataset.shape[0], -1)

reconstruction_errors = []

pca = IncrementalPCA()
pca.fit(dataset)

for i in ks:
    ipca = IncrementalPCA(n_components=i)
    features = ipca.fit_transform(dataset)
    image_pca = ipca.inverse_transform(features)

    # Calculate reconstruction error
    reconstruction_error = mean_squared_error(dataset, image_pca.reshape(dataset.shape))
    reconstruction_errors.append(reconstruction_error)

    filename = pcaPath + 'features_' + str(i) + '.txt'
    with open(filename, 'w') as f:
        json.dump(features.tolist(), f)
        print("Features stored in file for Principal components =", i)

print("PCA done")

modifiedPath = '.\\data\\size-500\\raster_mod\\'
pcaPath = '.\\data\\size-500\\PCA\\PCA_'
files = os.listdir(modifiedPath)
ks2 = [5, 20, 45, 70, 100, 140, 180, 230, 300, 360, 420, 520]

dataset = []
for file in files:
    image_bw = cv2.imread(modifiedPath + file, cv2.IMREAD_GRAYSCALE)
    dataset.append(image_bw)
dataset = np.array(dataset)
dataset = dataset.reshape(dataset.shape[0], -1)

reconstruction_errors2 = []

pca = IncrementalPCA()
pca.fit(dataset)

for i in ks2:
    ipca = IncrementalPCA(n_components=i)
    features = ipca.fit_transform(dataset)
    image_pca = ipca.inverse_transform(features)

    # Calculate reconstruction error
    reconstruction_error2 = mean_squared_error(dataset, image_pca.reshape(dataset.shape))
    reconstruction_errors2.append(reconstruction_error2)

    filename = pcaPath + 'features_' + str(i) + '.txt'
    with open(filename, 'w') as f:
        json.dump(features.tolist(), f)
        print("Features stored in file for Principal components =", i)

print("PCA done")


# Plot the reconstruction error
plt.plot(ks, reconstruction_errors, 'bo-', label = 'Dataset size-100')
plt.plot(ks2, reconstruction_errors2, 'ro-', label = 'Dataset size-500')
plt.legend()
plt.xlabel('Number of Principal Components')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error for PCA')
plt.savefig('Reconstruction Error.png')
plt.show()
