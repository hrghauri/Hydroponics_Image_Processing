import cv2
import numpy as np

image = cv2.imread('pic2.jpg')
#cv2.imshow('Input image', image)
#cv2.waitKey(0)
small = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
cv2.imshow('Small', small)
cv2.waitKey(0)

hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

lower_green = np.array([40,50,50])
upper_green = np.array([70,255,255])


mask = cv2.inRange(hsv, lower_green, upper_green)
res = cv2.bitwise_and(small, small, mask = mask)

cv2.imshow('Result', res)
cv2.waitKey(0)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
