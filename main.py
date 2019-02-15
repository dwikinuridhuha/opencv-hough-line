import cv2
import numpy as np

img = cv2.imread('src/img_test.jpg')
grs_img = np.copy(img)
abuAbu = cv2.cvtColor(grs_img, cv2.COLOR_RGB2GRAY)
cv2.imshow('result', abuAbu)
cv2.waitKey(0)