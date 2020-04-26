# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:08:35 2020

@author: Ambi
"""
import os
import argparse
import glob
import cv2
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-src", "--source", required=True,
	help="directory path of existing images")
ap.add_argument("-dst", "--destination", required=True,
	help="directory path to output directory of images")
args = vars(ap.parse_args())

imgs_list = glob.glob(args["source"] + "\\*\\*.jpg")
image_size = (512, 512)
print('Total images : ' + str(len(imgs_list)))

count = 0
for img in imgs_list:
    file_name = '%06d.jpg'%(count)
    count += 1
    #shutil.copyfile(img, args["destination"] + os.path.sep + file_name)
    image = cv2.imread(img)
    image = cv2.resize(image, image_size)
    cv2.imwrite(args["destination"] + os.path.sep + file_name, image)