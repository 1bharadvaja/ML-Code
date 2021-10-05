#proof of concept of loading up all the images in the data

import cv2
import numpy as np 
import glob

data = []
files = glob.glob('/Users/thema/OneDrive/Desktop/ML Code/png_data/training/LR/*.png')

for myFile in files:
	print(myFile)
	image = cv2.imread(myFile)
	data.append(image)
yennu = np.array(data)
print(yennu)
print(yennu.shape)