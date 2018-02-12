import cv2
import numpy as np

image = cv2.imread('pic.jpg')
#cv2.imshow('Input image', image)
#cv2.waitKey(0)
small = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
cv2.imshow('Small', small)
cv2.waitKey(0)

g = small.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

cv2.imshow('g', g)
cv2.waitKey(0)


gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray image', gray)
cv2.waitKey(0)

edged = cv2.Canny(small, 80, 200)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
