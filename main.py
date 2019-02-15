import cv2

img = cv2.imread('src/img_test.jpg')
cv2.imshow('result', img)
cv2.waitKey(0)