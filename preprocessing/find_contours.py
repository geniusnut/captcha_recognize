# https://stackoverflow.com/questions/48244328/copy-shape-to-blank-canvas-opencv-python
import numpy as np
import cv2

im = cv2.imread('RJOPCS_concat.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]