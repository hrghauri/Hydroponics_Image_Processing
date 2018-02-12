import cv2
import numpy as np

def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


def print_all_areas(contours):
    all_areas = get_contour_areas(contours)
    for area in all_areas:
        print("Area:" + str(area))


def label_contour_center(image, contours):

    for cnt in contours:
        if cv2.contourArea(cnt) > 0:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01']/M['m00'])
            print( "is closed : " + str(cv2.isContourConvex(cnt)))
            #cv2.circle(image,(cx,cy),10,(255,0,0),-1)
            cv2.putText(image, str(cv2.contourArea(cnt)), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    return image



image = cv2.imread('pic.jpg')
height, width = image.shape[:2]
#cv2.imshow('Input image', image)
#cv2.waitKey(0)
#small = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
small = cv2.resize(image, (int(width/3),int(height/3)), interpolation = cv2.INTER_AREA)

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

kernal = np.ones((5,5),np.float32)/25
smoothed = cv2.filter2D(res, -1, kernal)
cv2.imshow('Smoothed', smoothed)
cv2.waitKey(0)

kernal2 = np.ones((10,10),np.float32)



gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
#gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray image', gray)
cv2.waitKey(0)


opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernal2)
cv2.imshow('opening', opening)
cv2.waitKey(0)

edged = cv2.Canny(opening, 100,200)
cv2.imshow('Edged', edged)
cv2.waitKey(0)

blank_image = np.zeros((small.shape[0], small.shape[1], 3))


_, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(blank_image, contours, -1, (0,0,255),2)
cv2.imshow('Canny Edges After Contouring', blank_image)
cv2.waitKey()

print("Number of Contours found = " + str(len(contours)))

cv2.drawContours(small, contours, -1, (0,0,255),2)
cv2.imshow('Contours', small.copy())
cv2.waitKey()



print_all_areas(contours)

cv2.imshow('Contours with circles', label_contour_center(blank_image.copy(),contours))
cv2.waitKey()
