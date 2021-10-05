import argparse
import numpy as np
from numpy import random
from patchify import patchify
import os
import h5py
import pydicom
import scipy.misc
import imageio
import PIL 
from PIL import Image
import cv2
import glob


data = []
data_files = glob.glob('/Users/thema/OneDrive/Desktop/ML_Code/png_data/training/LR/*.png')
label = []

for myData in data_files:
	basename = os.path.basename(myData)
	img = Image.open(myData)
	width, height = img.size
	img1 = img.resize((width*2, height*2))
	patch_x = np.random.randint(0, width*2 - 64)
	patch_y = np.random.randint(0, height*2 - 64)
	left = patch_x 
	top = patch_y
	right = patch_x + 64
	bottom = patch_y + 64
	img1 = img1.crop((left, top, right, bottom))
	img1_array = np.asarray(img1)
	data.append(img1_array)
	hr_img = Image.open('/Users/thema/OneDrive/Desktop/ML_Code/png_data/training/HR/' + basename)
	hr_img = hr_img.crop((left, top, right, bottom))
	hr_array = np.asarray(hr_img)
	label.append(hr_array)
data = np.asarray(data)
label = np.asarray(label)
print(data)
print(label)
print("data", data.shape)
print("label",)