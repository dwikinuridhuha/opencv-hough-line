import cv2
import numpy as np

img = cv2.imread('src/img_test.jpg')
grs_img = np.copy(img)
abuAbu = cv2.cvtColor(grs_img, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(abuAbu, (5, 5), 0)
canny = cv2.Canny(blur, 50, 150)
cv2.imshow('result', canny)
cv2.waitKey(0)