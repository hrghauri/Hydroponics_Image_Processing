import cv2
import numpy as np

image = cv2.imread('pic.jpg')
height, width = image.shape[:2]


original = cv2.resize(image, (int(width/3),int(height/3)), interpolation = cv2.INTER_AREA)
cv2.imshow('Original RGB', original)
cv2.waitKey(0)


gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', gray)
cv2.waitKey(0)

sobelx = cv2.Sobel(original,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(original,cv2.CV_64F,0,1,ksize=5)


cv2.imshow('sobelx', sobelx)
cv2.waitKey(0)

cv2.imshow('sobely', sobely)
cv2.waitKey(0)

edged = cv2.Canny(original, 100,200)
cv2.imshow('edged', edged)
cv2.waitKey(0)

#hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
#cv2.imshow('HSV image', hsv)
#cv2.waitKey(0)

#rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#cv2.imshow('RGB image', rgb )
#cv2.waitKey(0)
