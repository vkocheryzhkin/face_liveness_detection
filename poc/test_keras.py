from __future__ import print_function
import os 
from tqdm import tqdm
import cv2
import numpy as np
import argparse

# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
# from keras.models import model_from_yaml

# # load YAML and create model
# yaml_file = open('model.yaml', 'r')
# loaded_model_yaml = yaml_file.read()
# yaml_file.close()
# loaded_model = model_from_yaml(loaded_model_yaml)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")


# im_w = 150
# im_h = 150

# # live_file = "/home/vladimir/Work/face_liveness_detection_data/live1_of/100.png"
# # live_file = "/home/vladimir/Work/face_liveness_detection_data/fake1_of/200.png"
# img = cv2.imread(live_file, cv2.IMREAD_COLOR)

# X = np.array([img], dtype=float)
# X /= 255
# sample = X.reshape((-1, im_h, im_w, 3))

# classes = loaded_model.predict(sample)
# print(classes)

def process(left, right):
  print(left, right)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Process video')
  parser.add_argument('-l', '--leftimage', help="Specify left image")
  parser.add_argument('-r','--rightimage', help="Specify right image")
  # parser.add_argument('-i','--input_of_image', help="Specify right image")
  args = parser.parse_args()
  process(args.leftimage, args.rightimage)