import cv2
import numpy as np
import math

def rotate_image(mat, angle):
  height, width = mat.shape[:2]
  image_center = (width / 2, height / 2)

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

  radians = math.radians(angle)
  sin = math.sin(radians)
  cos = math.cos(radians)
  bound_w = int((height * abs(sin)) + (width * abs(cos)))
  bound_h = int((height * abs(cos)) + (width * abs(sin)))

  rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
  rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat

scalexy = 0.5
frame_skip = 5
name = 'live1'
ext = '.MOV'

vidcap = cv2.VideoCapture(name + ext)

success,image = vidcap.read()
image = rotate_image(image, -90)
image = cv2.resize(image, (0,0), fx=scalexy, fy=scalexy)

count = 0
while success: # and count < 3:
  image_path = "%s/%04d.png" %(name, count)
  # print(image_path)
  # image_stamp = "%04d" % count
  if count % frame_skip == 0:
    cv2.imwrite(image_path, image)     # save frame as JPEG file        
  success,image = vidcap.read()
  if success:
    image = rotate_image(image, -90)
    image = cv2.resize(image, (0,0), fx=scalexy, fy=scalexy) 
  print('Read a new frame: ', success)
  count += 1