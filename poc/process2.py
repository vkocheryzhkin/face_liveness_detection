import cv2 as cv
import numpy as np
import os, os.path
import itertools  

def get_flow(left, right):
    frame1 = cv.imread(left)
    frame2 = cv.imread(right)
    # frame1 = cv.imread("/home/vladimir/Work/face_liveness_detection_data/live1/0000.jpg")
    # frame2 = cv.imread("/home/vladimir/Work/face_liveness_detection_data/live1/0060.jpg")

    # frame1 = cv.imread("/home/vladimir/Work/face_liveness_detection_data/0001.png")
    # frame2 = cv.imread("/home/vladimir/Work/face_liveness_detection_data/0051.png")

    # frame1 = cv.imread("/home/vladimir/Work/face_liveness_detection_data/v_photo/111.jpg")
    # frame2 = cv.imread("/home/vladimir/Work/face_liveness_detection_data/v_photo/222.jpg")

    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)

    return hsv

def process(images_dir):
    res_dir = images_dir + "_of"
    files = [os.path.join(images_dir, name) for name in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, name))]
    ids = list(range(0, len(files)))
    count = 0
    for t in itertools.combinations(ids,2):
        left = files[0]
        right = files[1]
        hsv = get_flow(left, right)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        filename = "{}/{}.png".format(res_dir, count)
        cv.imwrite(filename, bgr)
        count += 1

if __name__ == "__main__":
    path = "/home/vladimir/Work/face_liveness_detection_data/fake1"
    process(path)