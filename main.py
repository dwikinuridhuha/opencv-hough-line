import cv2
import numpy as np
import matplotlib.pylab as plt

def tampilkan_garis(img, garis):
    line_img = np.zeros_like(img)
    if garis is not None:
        for line in garis:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img

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

crop_img = membuat_segi_tiga_bertua(canny)
lines = cv2.HoughLinesP(crop_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
garis_gmbr = tampilkan_garis(imgCpy, lines)
combo_img = cv2.addWeighted(imgCpy, 0.8, garis_gmbr, 1, 1)

# plt.imshow(canny)
# plt.show()
cv2.imshow("hasil", combo_img)
cv2.waitKey(0)