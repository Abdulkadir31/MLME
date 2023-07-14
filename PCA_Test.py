import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread
import json
import os

pca = PCA()


modifiedPath = '.\\data\\size-100\\raster_mod\\'
pcaPath = '.\\data\\size-100\\PCA\\PCA_'
files = os.listdir(modifiedPath)
ks = [3, 5, 7, 10, 25, 40, 60, 80, 108]
#ks = [5, 20, 45, 70, 100, 140, 180, 230, 300, 360, 420, 520]


# for i in ks:
# 	if not os.path.exists(pcaPath+str(i)):
# 		os.makedirs(pcaPath+str(i))
# 		print("Directory created at "+pcaPath+str(i))

dataset = []
for file in files:
	image_bw = cv2.imread(modifiedPath+file, cv2.IMREAD_GRAYSCALE)
	dataset.append(image_bw)
dataset = np.array(dataset)
dataset = dataset.reshape(dataset.shape[0],-1)

pca.fit(dataset)
	
for i in ks:
	ipca = IncrementalPCA(n_components=i)
	# print(len(ipca.fit_transform([temp])))
	# image_pca = ipca.inverse_transform(ipca.fit_transform(image_bw))
	features = ipca.fit_transform(dataset)
	print(features.shape)
	# plt.imshow(image_recon,cmap = plt.cm.gray)
	# cv2.imwrite(pcaPath+str(i)+'\\'+file,image_pca)
	image_pca = ipca.inverse_transform(features)

	filename = pcaPath+'features_'+str(i)+'.txt' 
	with open(filename , 'w') as f:
		json.dump(features.tolist(), f)
		print("Features stored in file for Principal components = "+str(i))



	# print(image_pca.shape)
print("PCA done")  

# Getting the cumulative variance

var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
# print(var_cumu)
# How many PCs explain 95% of the variance?
k = np.argmax(var_cumu>95)
print("Number of components explaining 95% variance: "+ str(k)) 

pca_64 = PCA(n_components=108)
pca_64.fit(dataset)

plt.grid()
plt.plot(np.cumsum(pca_64.explained_variance_ratio_ * 100))
plt.xlabel('Number of Principal Components')
plt.ylabel('Percentage Explained variance')
plt.savefig('Scree plot.png')
plt.show()

# Used the below code only once for saving the figures

def plot_at_k(k,file=None):
    ipca = IncrementalPCA(n_components=k)
    image_pca = ipca.inverse_transform(ipca.fit_transform(dataset))
    plt.imshow(image_pca[file].reshape(64,64),cmap = plt.cm.gray)
    # cv2.imwrite(pca_path+str(k)+'\\'+file,image_pca)

ks = [3, 5, 7, 10, 25, 50, 80, 100, 108]
#ks = [5, 20, 45, 70, 100, 140, 180, 230, 300, 360, 420, 520]
for j in range(108):
	plt.figure(figsize=[15,15])

	for i in range(9):
	    plt.subplot(3,3,i+1)
	    plot_at_k(ks[i],j)
	    plt.title("Components: "+str(ks[i]))

	plt.subplots_adjust(wspace=0.2, hspace=0.4)
	plt.savefig('./data/PCA_Images/size-100/'+str(j)+'.png')
	plt.close()
	print('Saved Image '+str(j))
# plt.show()
print('All pictures saved')
