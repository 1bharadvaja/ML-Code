import argparse
import os
import h5py
import PIL
from PIL import Image
from cyclegan import CycleGAN
from the_parser import training_parser,testing_parser

import numpy as np

def main():
    args = testing_parser().parse_args()

    name = args.name
    model = args.model
    
    #File paths
    train_dir = os.path.join('train/', name)

    #f = h5py.File('./test_v2nds_2D.h5', 'r')
    #test_data = f.get('LR')
    #test_label = f.get('SR')

    x = Image.open('/Users/thema/OneDrive/Desktop/ML_Code/png_data/testing/LR/MarginDefault_0014bscan_0142.png')
    x1 = np.asarray(x)
    test_data = x1[np.newaxis, :, :, np.newaxis]

    y = Image.open('/Users/thema/OneDrive/Desktop/ML_Code/png_data/testing/HR/MarginDefault_0014bscan_0142.png')
    y1 = np.asarray(y)
    test_label = y1[np.newaxis, :, :, np.newaxis]


    args.w = test_data.shape[1]
    args.h = test_data.shape[2]
    args.c = test_data.shape[3]
    args.ow = test_label.shape[1]
    args.oh = test_label.shape[2]
    args.oc = test_label.shape[3]

    print(test_data,test_label)
    cyclegan = CycleGAN(args, False, None)

    ## b -> label  a -> data
    b2a, a2b, aba, bab = cyclegan.test(test_data,test_label)  # return array
    print("b2a", b2a)
    print(test_label)
    squeezed = np.squeeze(b2a)
    img = Image.fromarray(squeezed)
    img = img.convert("L")
    img.save('b2a3.png')
    print(a2b.shape)
    print(aba.shape)
    print(bab.shape)

if __name__ == '__main__':
    main()
