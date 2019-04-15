import cv2
import numpy as np

video = cv2.VideoCapture("src/1.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        video = cv2.VideoCapture("src/1.mp4")
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([18, 94, 140])
    up_yellow = np.array([48, 255, 255])
    mask = cv2.inRange(hsv, low_yellow, up_yellow)
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=100)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(25)
    if(key == 27):
        break
video.release()
cv2.destroyWindow()


########### untuk Photo
# img = cv2.imread("src/img_test.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edge = cv2.Canny(gray, 75, 150)

# lines = cv2.HoughLinesP(edge, 1, np.pi/180, 50)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (100, 255, 100), 3)

# cv2.imshow("edge", edge)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyWindow()