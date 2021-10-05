
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
from cyclegan import CycleGAN
from the_parser import training_parser


os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():	
	args = training_parser().parse_args()

	name = args.name
	restore = args.restore
	restore_ckpt = True if restore else False

	data = []
	data_files = glob.glob('/Users/thema/Desktop/ML_Code/png_data/training/LR/*.png')
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
		hr_img = Image.open('/Users/thema/Desktop/ML_Code/png_data/training/HR/' + basename)
		hr_img = hr_img.crop((left, top, right, bottom))
		hr_array = np.asarray(hr_img)
		label.append(hr_array)

	data = np.asarray(data)
	label = np.asarray(label)
	data1 = data[:, :, :, np.newaxis]
	label1 = label[:, :,  :, np.newaxis]
	print(data1.shape, label1.shape)


	args.w = data1.shape[1]
	args.h = data1.shape[2]
	args.c = data1.shape[3]

	args.ow = label1.shape[1]
	args.oh = label1.shape[2]
	args.oc = label1.shape[3]
	print(data)
	#File paths
	train_dir = os.path.join('Network/', name)

	cyclegan = CycleGAN(args, True, restore_ckpt)	
	cyclegan.train(data1, label1)
if __name__ == '__main__':
    main()