import cv2
import numpy as np
import matplotlib.pylab as plt

def canny(img):
    self

def membuat_segi_tiga_bertua(img):
    tinggi = img.shape[0]
    polygon = np.array([
    [(200, tinggi), (1100, tinggi), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    mask_img = cv2.bitwise_and(img, mask)
    return mask_img

img = cv2.imread('src/img_test.jpg')
imgCpy = np.copy(img)
abuAbu = cv2.cvtColor(imgCpy, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(abuAbu, (5, 5), 0)
canny = cv2.Canny(blur, 50, 150)

# plt.imshow(canny)
# plt.show()
crop_img = membuat_segi_tiga_bertua(canny)
cv2.imshow("hasil", crop_img)
cv2.waitKey(0)