# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:24:35 2020

@author: Ambi
"""

import argparse
import glob
import os
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image

class FaceCrop:
    
    def __init__(self, required_size=(72,80)):
        self.image_size = required_size
        self.detector = MTCNN()

    def crop_images(self, src, dst):
        images_list = glob.glob(src + '*.jpg')
        print(len(images_list))
    
        if not os.path.exists(dst):
            os.mkdir(dst)
            
        for image in images_list:
            img = cv2.imread(image)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            file_name = image.split('/')[-1]
            print(file_name)
            rects = self.detector.detect_faces(img)
            for i, rect in enumerate(rects):
                x1, y1, width, height = rect['box']
                x2, y2 = x1 + width, y1 + height
            
                if y1 < 0 or y2 >= img.shape[0] or x1 < 0 or x2 >= img.shape[1]:
                    print(str((x1, y1, x2, y2)) + ' is beyond image of size: ' + str(img.shape) 
                          + ' for file ' + file_name)
                    if x1 < 0:
                        x1 = max(x1, 0)
                    if y1 < 0:
                        y1 = max(y1, 0)
                    if x2 >= img.shape[1]:
                        x2 = min(x2, img.shape[1])
                    if y2 >= img.shape[0]:
                        y2 = min(y2, img.shape[0])
                
                face = img[y1:y2, x1:x2]
                face = cv2.resize(face, (72, 80))
                
                dst_file_name = dst + str(i) + '_' + file_name
                print('Writing cropped face to: ' + dst_file_name)
                cv2.imwrite(dst_file_name, face)
                

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", "--source", required=True,
    	help="Folder of images fow which faces need to be cropped")
    ap.add_argument("-dst", "--destination", required=True,
    	help="Folder of images fow which faces need to be cropped")

    args = vars(ap.parse_args())    
    face_crop = FaceCrop()
    face_crop.crop_images(args["source"], args["destination"])