# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 01:09:29 2020

@author: Ambi
"""

import os
import glob
import numpy as np
import cv2
import time
import argparse
import pickle

from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

from keras import Model
from keras.utils import to_categorical
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from keras.layers import GlobalAveragePooling2D

import nmslib

class FeatureExtractor:
    def __init__(self, model_type, debug, image_size=(224, 224, 3)):
        self.image_size = image_size
        self.model_type = model_type
        self.model = VGGFace(model='resnet50')
        self.detector = MTCNN()
        self.debug = debug
    
    def get_feature_extraction_model(self):
        model = VGGFace(model=self.model_type, include_top=False
                        , input_shape=self.image_size, pooling='avg')
        output = model.get_layer('conv5_3').output
        output = GlobalAveragePooling2D()(output)
        feature_model = Model(inputs=model.input, outputs=output)
        return feature_model
    
    def extract_face(self, image_file, required_size=(224,224)):
        image = plt.imread(image_file)
        start_time = time.time()
        rects = self.detector.detect_faces(image)
        end_time = time.time()
        if self.debug:
            print('Detected %d faces in %s at [%.3f] seconds' % (len(rects), image_file, (end_time - start_time)))
        if rects is None or len(rects) == 0:
            if self.debug:
                print('MTCNN did not detect face for: ' + image_file.split('\\')[-1])
            return None
        faces = []
        for rect in rects:
            x1, y1, width, height = rect['box']
            x2, y2 = x1 + width, y1 + height
        
            if y1 < 0 or y2 >= image.shape[0] or x1 < 0 or x2 >= image.shape[1]:
                if self.debug:
                    print(str((x1, y1, x2, y2)) + ' is beyond image of size: ' + str(image.shape) 
                      + ' for file ' + image_file.split('\\')[-1])
                x1 = min(x1, 0)
                y1 = min(y1, 0)
                x2 = max(x2, image.shape[1])
                y2 = max(y2, image.shape[0])
                
            face = image[y1:y2, x1:x2]
            face = Image.fromarray(face)
            face = face.resize(required_size)
            face = np.asarray(face)
            faces.append(face)
        return faces
    
    def get_face_embeddings(self, images_list):
        count = 0
        start_time = time.time()
        faces_list = []
        iter_start_time = time.time()
        for image_file in images_list:
            faces = self.extract_face(image_file)
            if faces is None or len(faces) <= 0:
                #print('Did not find face in image: ' + image_file.split('\\')[-1])
                continue
            faces_list.append(faces)
            count += 1
            if count % 100 == 0:
                iter_end_time = time.time()
                print('Extracted %d faces in [%.3f] seconds ' % (count, (iter_end_time - iter_start_time)))
                iter_start_time = time.time()
        end_time = time.time()
        print('Extracted faces in %d images at [%.3f] seconds' % (len(images_list), (end_time - start_time)))
        
        count = 0
        face_image_map = dict()
        all_faces = []
        for id, faces in enumerate(faces_list):
            for face in faces:
                face_image_map[count] = id
                all_faces.append(face)
                count += 1
        print('Total %d faces detected' % len(all_faces))
        start_time = time.time()
        faces = np.asarray(all_faces, 'float32')
        faces = utils.preprocess_input(faces, version=2)
        end_time = time.time()
        print('Preprocessed faces in [%.3f] seconds: ' % (end_time - start_time))
        model = self.get_feature_extraction_model()
        
        start_time = time.time()
        face_embeddings = model.predict(faces)
        end_time = time.time()
        print('Extracted features in [%.3f] seconds' % (end_time - start_time))
        
        return face_embeddings, face_image_map
    
    def index_save_features(self, face_embedding, face_image_map):
        feature_vectors = np.array(face_embedding).astype(np.float32)
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(feature_vectors)
        index.createIndex({'post': 2}, print_progress=True)
        
        print('Saving hnsw index')
        index.saveIndex('FacialFeatures.hnsw')
        
        print('Saving face to image mapping disctionary')
        file = open('face_to_image.pkl', 'wb')
        pickle.dump(face_image_map, file)
        file.close()
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_type", "--model_type", required=True,
    	help="Model type(vgg16, resnet50, sesnet50)")
    ap.add_argument("-debug", "--debug", required=True,
    	help="True, if logs to be printed, else False")
    ap.add_argument("-images", "--images", required=True,
    	help="Directory of images for which facial features to be extracted")
    
    args = vars(ap.parse_args())
    
    featureExtractor = FeatureExtractor(args["model_type"], args["debug"])
    imgs_list = glob.glob(args["images"] + "*.jpg")
    face_embedding, face_image_map = featureExtractor.get_face_embeddings(imgs_list[:2])
    featureExtractor.index_save_features(face_embedding, face_image_map)
    