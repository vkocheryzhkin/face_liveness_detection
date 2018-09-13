import cv2 as cv
import numpy as np
import math

def rotate_image(mat, angle):
  height, width = mat.shape[:2]
  image_center = (width / 2, height / 2)

  rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1)

  radians = math.radians(angle)
  sin = math.sin(radians)
  cos = math.cos(radians)
  bound_w = int((height * abs(sin)) + (width * abs(cos)))
  bound_h = int((height * abs(cos)) + (width * abs(sin)))

  rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
  rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

  rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat

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