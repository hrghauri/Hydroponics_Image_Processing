import cv2
import numpy as np



image = cv2.imread('Test.png')
height, width = image.shape[:2]
resized_image = cv2.resize(image, (int(width/3),int(height/3)), interpolation = cv2.INTER_AREA)
cv2.imshow('Image', resized_image)
cv2.waitKey(0)

hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

lower_green = np.array([40,50,50])
upper_green = np.array([70,255,255])

mask = cv2.inRange(hsv_image, lower_green, upper_green)
result = cv2.bitwise_and(resized_image, resized_image, mask = mask)
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

kernal = np.ones((15,15),np.float32)/225
#kernal = np.ones((10,10),np.float32)/100
#kernal = np.ones((5,5),np.float32)/25
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernal)

edged = cv2.Canny(opening, 100,200)

blank_image = np.zeros((resized_image.shape[0], resized_image.shape[1], 3))
_, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(blank_image, contours, -1, (0,0,255),2)
cv2.imshow('Canny Edges After Contouring', blank_image)
cv2.waitKey()

cv2.drawContours(resized_image, contours, -1, (0,0,255),2)
cv2.imshow('Contours', resized_image.copy())
cv2.waitKey()


print("Number of Contours found = " + str(len(contours)))


for cnt in contours:
    if cv2.contourArea(cnt) > 0:
        area = cv2.contourArea(cnt)
        print("Area " + str(area))
        print(str(cv2.isContourConvex(cnt)))

        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(blank_image, str(cv2.contourArea(cnt)), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

cv2.imshow('Labeled with areas', blank_image)
cv2.waitKey()
