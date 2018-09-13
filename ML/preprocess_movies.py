import cv2 as cv
import numpy as np
import math
import argparse
import dlib
from cv_common import rotate_image

FACE_SIZE = 150

def process(input, output, type, frame_skip):
  detector = dlib.get_frontal_face_detector()
  sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

  vidcap = cv.VideoCapture(input)

  success,image = vidcap.read()
  image = rotate_image(image, -90)
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

  count = 0
  while success:
    image_path = "%s/%04d" %(output, count)
    if count % frame_skip == 0:
      dets = detector(image)
      if(len(dets) > 0):
        d = dets[0]
        shape = sp(image, d)
      else:
        print("face not found")
      dlib.save_face_chip(image, shape, image_path, size=FACE_SIZE, padding=0.25)
    success,image = vidcap.read()
    if success:
      image = rotate_image(image, -90)
      image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    count += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Process video')
  parser.add_argument('-i', '--inputmovie', help="Specify the input movie")
  parser.add_argument('-o', '--outputdir', help="Specify the output directory")
  parser.add_argument("-s", '--skipframes', help="skip frames", default=2, type=int)
  parser.add_argument('--fraud', dest='type', action='store_false')
  parser.add_argument('--live', dest='type', action='store_true')
  parser.set_defaults(type=True)
  args = parser.parse_args()
  if not args.inputmovie:
    parser.error('Please specify an input movie')
  if not args.outputdir:
    parser.error('Please specify an output directory')
  
  process(args.inputmovie, args.outputdir, args.type, args.skipframes)
