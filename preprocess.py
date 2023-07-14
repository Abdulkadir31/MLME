import os
import cv2
import imutils

# Declaring the constants
originalPath = '.\\data\\size-100\\raster\\'
modifiedPath = '.\\data\\size-100\\raster_mod\\'
color = [0,0]
desired_size = 64


files = os.listdir(originalPath)
for file in files:

# Converting the image to black and white	
	img_rgb = cv2.imread(originalPath+file)
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)[1]
	
	h = img_bw.shape[0]
	w = img_bw.shape[1]
	
# Reducing the size of the image
	if h >= w:
		img_bw = imutils.resize(img_bw, height = 64)
	else:
		img_bw = imutils.resize(img_bw, width = 64)

# Adding padding to image based on dimensions
	h = img_bw.shape[0]
	w = img_bw.shape[1]
	left, right, top, bottom = 0, 0, 0, 0
	if w <= h:
		delta_w = 64 - w
		left, right = delta_w//2, delta_w-(delta_w//2)
	else:
		delta_h = 64 - h
		top, bottom = delta_h//2, delta_h-(delta_h//2)

	img_bw = cv2.copyMakeBorder(img_bw, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# Saving the image
	cv2.imwrite(modifiedPath+file,img_bw)

print('All Images are Converted')
