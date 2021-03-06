import cv2
import numpy as np
import matplotlib.pylab as plt

def kordinat(img, garis_params):
    slope, intersep = garis_params
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intersep)/slope)
    x2 = int((y2 - intersep)/slope)
    return np.array([x1, y1, x2, y2])

def avg_slope_intercept(img, garis):
    kiri_fit = []
    kanan_fit = []
    for line in garis:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intersep = params[1]
        if slope < 0:
            kiri_fit.append((slope, intersep))
        else:
            kanan_fit.append((slope, intersep))
    kiri_avg = np.average(kiri_fit, axis=0)
    kanan_avg = np.average(kanan_fit, axis=0)
    garis_kiri = kordinat(img, kiri_avg)
    garis_kanan = kordinat(img, kanan_avg)
    return np.array([garis_kiri, garis_kanan])

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

# img = cv2.imread('src/img_test.jpg')
# imgCpy = np.copy(img)
# abuAbu = cv2.cvtColor(imgCpy, cv2.COLOR_RGB2GRAY)
# blur = cv2.GaussianBlur(abuAbu, (5, 5), 0)
# canny = cv2.Canny(blur, 50, 150)

# crop_img = membuat_segi_tiga_bertua(canny)
# lines = cv2.HoughLinesP(crop_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# garis_avg = avg_slope_intercept(imgCpy, lines)
# garis_gmbr = tampilkan_garis(imgCpy, garis_avg)
# combo_img = cv2.addWeighted(imgCpy, 0.8, garis_gmbr, 1, 1)

# # plt.imshow(canny)
# # plt.show()
# cv2.imshow("hasil", combo_img)
# cv2.waitKey(0)

cap = cv2.VideoCapture("src/3.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(abuAbu, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    crop_img = membuat_segi_tiga_bertua(canny)
    lines = cv2.HoughLinesP(crop_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    garis_avg = avg_slope_intercept(frame, lines)
    garis_gmbr = tampilkan_garis(frame, garis_avg)
    combo_img = cv2.addWeighted(frame, 0.8, garis_gmbr, 1, 1)
    cv2.imshow("result", combo_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()