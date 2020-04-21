# https://stackoverflow.com/questions/48244328/copy-shape-to-blank-canvas-opencv-python
#!/usr/bin/python3
# 2018.01.14 09:48:15 CST
# 2018.01.14 11:39:03 CST

import numpy as np
import cv2

im = cv2.imread('s9jtq.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]


## this contour is a 3D numpy array
cnt = contours[2]
res = cv2.drawContours(im,[cnt],0,(255,0,0), -1)
cv2.imwrite("contours.png", res)

## Method 1: crop the region
x,y,w,h = cv2.boundingRect(cnt)
croped = res[y:y+h, x:x+w]
cv2.imwrite("croped.png", croped)

## Method 2: draw on blank
# get the 0-indexed coords
offset = cnt.min(axis=0)
cnt = cnt - cnt.min(axis=0)
max_xy = cnt.max(axis=0) + 1
w,h = max_xy[0][0], max_xy[0][1]
# draw on blank
canvas = np.ones((h,w,3), np.uint8)*255
cv2.drawContours(canvas, [cnt], -1, (255,0,0), -1)
cv2.imwrite("canvas.png", canvas)