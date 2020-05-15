# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:31:34 2020

@author: Ambi
"""

import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras import Model
from keras.layers import GlobalAveragePooling2D
import os
from glob import glob
import argparse
import nmslib
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import time

class FeatureExtractor:
    def __init__(self, model_type, debug, image_size=(224, 224, 3)):
        self.image_size = image_size
        self.model_type = model_type
        self.debug = debug

    def get_feature_extraction_model(self):
        if self.model_type == "vgg16":
            model = VGGFace(model=self.model_type, include_top=False
                            , input_shape=self.image_size, pooling='avg')
            output = model.get_layer('conv5_3').output
            output = GlobalAveragePooling2D()(output)
            feature_model = Model(inputs=model.input, outputs=output)
        elif self.model_type == "resnet50":
            feature_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        return feature_model
    
    def get_deep_feature(self, x):
    
        feature_model = self.get_feature_extraction_model()
        x = utils.preprocess_input(x, version=1)  # or version=2
        x = feature_model.predict(x)
    
        return x

    def extract_features(self, src, dst):
        face_id_map = dict()
        face_list = list()
        image_count = 0
        start_time = time.time()
        for id in range(20):
            for category in ['train', 'val', 'test']:
                print('In %s folder of %d' % (category, id))
                images = glob(os.path.join(src, category, str(id), '*.png'))
                print(len(images))
                for image in images:
                    img = plt.imread(image)
                    img = Image.fromarray((img * 255).astype(np.uint8))
                    img = np.array(img.resize((224, 224)))
                    if img.ndim == 2:
                        img = np.expand_dims(img, -1)
                        img = np.concatenate((img, img, img), axis=-1)
                    face_list.append(img)
                    path = category + '/' + str(id) + '/' + image.split('\\')[-1]
                    
                    face_id_map[image_count] = path
                    image_count += 1
                    
        faces = np.asarray(face_list)
        faces = faces.astype(np.float64)
        end_time = time.time()
        print('Finished reading ' + str(faces.shape) + ' images in [%.3f] seconds' % ((end_time - start_time)))
        start_time = time.time()
        face_features = self.get_deep_feature(faces)
        end_time = time.time()
        print('Extracted features in [%.3f] seconds' % ((end_time - start_time)))
        return face_features, face_id_map
    
    def save_index_features(self, face_features, face_id_map, dst):
        feature_vectors = np.array(face_features).astype(np.float32)
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(feature_vectors)
        index.createIndex({'post': 2}, print_progress=True)
        
        output_folder = dst + '/MiniCelebA_'
        hnsw_index_path = output_folder + args["model_type"] + '_FacialFeatures.hnsw'
        print('Saving hnsw index @ : ' + hnsw_index_path)
        index.saveIndex(hnsw_index_path)
        
        face_image_path = output_folder + args["model_type"]+'_face_to_image.pkl'
        print('Saving face to image mapping disctionary @: ' + face_image_path)
        file = open(face_image_path, 'wb')
        pickle.dump(face_id_map, file)
        file.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_type", "--model_type", required=True,
    	help="Model type(vgg16, resnet50, sesnet50)")
    ap.add_argument("-debug", "--debug", required=True,
    	help="True, if logs to be printed, else False")
    ap.add_argument("-src", "--src", required=True,
    	help="Directory of images for which facial features to be extracted")
    ap.add_argument("-dst", "--dst", required=True,
    	help="Directory of features and images to be stored")
    args = vars(ap.parse_args())
    
    feature_extractor = FeatureExtractor(args["model_type"], args["debug"])
    face_features, face_id_map = feature_extractor.extract_features(args["src"], args["dst"])
    feature_extractor.save_index_features(face_features, face_id_map, args["dst"])