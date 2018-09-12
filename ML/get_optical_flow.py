import cv2 as cv
import numpy as np
import os, os.path
import itertools
import argparse

def get_flow(left, right):
  frame1 = cv.imread(left)
  frame2 = cv.imread(right)

  prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
  hsv = np.zeros_like(frame1)
  hsv[...,1] = 255

  next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
  flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
  hsv[...,0] = ang*180/np.pi/2
  hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)

  return hsv

def process(inputdir, output, type):
  files = [os.path.join(inputdir, name) for name in os.listdir(inputdir) if os.path.isfile(os.path.join(inputdir, name))]
  ids = list(range(0, len(files)))
  count = 0
  for t in itertools.combinations(ids, 2):
    left = files[0]
    right = files[1]
    hsv = get_flow(left, right)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    filename = "{}/{}.png".format(output, count)
    cv.imwrite(filename, bgr)
    count += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Process frames')
  parser.add_argument('-i', '--inputdir', help="Specify the input frames folder")
  parser.add_argument('-o', '--outputdir', help="Specify the output folder")
  parser.add_argument('--fraud', dest='type', action='store_false')
  parser.add_argument('--live', dest='type', action='store_true')
  parser.set_defaults(type=True)
  args = parser.parse_args()
  if not args.inputdir:
    parser.error('Please specify an input frames folder')
  if not args.outputdir:
    parser.error('Please specify an output folder')
  
  process(args.inputdir, args.outputdir, args.type)