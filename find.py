#finding whicch images aren't the size we want

import cv2
import numpy as np 
import glob
import PIL
from PIL import Image

data = []
files = glob.glob('/Users/thema/OneDrive/Desktop/ML Code/png_data/training/HR/*.png')

for myFile in files:
    ddata = Image.open(myFile)
    print(myFile)
    print(ddata.size)