import numpy as np
import math
import cv2
import random

# LOAD IMAGE
img = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_A01_w1_9E84F49F-1B25-4E7E-8040-D1BB2D7E73EA.tif')
grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("grayscale", grayscale)
cv2.waitKey(0)

#
# # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(60, 60))
# # filtered = clahe.apply(grayscale)
# # cv2.imshow("filtered", filtered)
# # cv2.waitKey(0)
#
# equ = cv2.equalizeHist(grayscale)
# cv2.imshow("non-CLAHE", equ)
# cv2.waitKey(0)
# ret, thresh = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("Otsu's binarization", thresh)
# cv2.waitKey(0)
#
# # remove = cv2.medianBlur(diff, 5)
# # cv2.imshow("remove noise", remove)
# # cv2.waitKey(0)
#
# NOISE REMOVAL
kernel = np.ones((2,2), np.uint8)
#
# # inverse = (255 - remove)
#
# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
#
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 2
# params.maxArea = 100
#
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
#
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.3
#
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01
#
# # Create a detector with the parameters
# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3:
#     detector = cv2.SimpleBlobDetector(params)
# else:
#     detector = cv2.SimpleBlobDetector_create(params)
#
# # Detect blobs.
# keypoints = detector.detect(equ)
#
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(equ, keypoints, np.array([]), (0, 0, 255),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# #
# # # Show keypoints
# # cv2.imshow("Keypoints", im_with_keypoints)
# # cv2.waitKey(0)
#
# for point in keypoints:
#     point.class_id = 99
#     # print (point.pt)
#     # print (point.class_id)
#
# inverse = cv2.bitwise_not(equ)
# canny = cv2.Canny(inverse, 100, 200)
# # cv2.imshow("edges", canny)
# # cv2.waitKey(0)
# canny2 = cv2.copyMakeBorder(canny, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
# for point in keypoints:
#     int_point = (int(point.pt[0]), int(point.pt[1]))
#     cv2.floodFill(inverse, canny2, int_point, (0,0,0))
# # cv2.imshow("filled", inverse)
# # cv2.waitKey(0)
#
# sure_fg = cv2.erode(inverse, kernel, iterations=2)
# # cv2.imshow("eroded", sure_fg)
# # cv2.waitKey(0)
#
# sure_bg = cv2.dilate(sure_fg, kernel, iterations=3)
# # cv2.imshow("dilated", sure_bg)
# # cv2.waitKey(0)
# print (len(sure_bg.shape))
# cv2.imshow("before inverse", sure_bg)
# cv2.waitKey(0)
#
# inversedAgain = cv2.bitwise_not(sure_bg)
# cv2.imshow("iv", inversedAgain)
# cv2.waitKey(0)
# cv2.imshow("sure_fg", sure_fg)
# cv2.waitKey(0)
#
# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
#
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 100
# params.maxArea = 9999
#
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.01
#
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01
#
# # Create a detector with the parameters
# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3:
#     detector2 = cv2.SimpleBlobDetector(params)
# else:
#     detector2 = cv2.SimpleBlobDetector_create(params)
#
# # Detect blobs.
# keypoints2 = detector2.detect(inversedAgain)
# totalKeypoints = detector2.detect(sure_bg)
#
# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints2 = cv2.drawKeypoints(sure_fg, keypoints2, np.array([]), (0, 0, 255),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# im_with_keypoints3 = cv2.drawKeypoints(inversedAgain, totalKeypoints, np.array([]), (0, 0, 255),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# # Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints2)
# cv2.waitKey(0)
# cv2.imshow("Keypoints", im_with_keypoints3)
# cv2.waitKey(0)
#
# for key in keypoints2:
#     if key not in totalKeypoints:
#         totalKeypoints.append(key)
#
# print (len(totalKeypoints))

trial = cv2.imread('test_file.png')
# trial = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_C16_w1_62DED874-8F2F-454A-94A8-70897603EED7.tif')
trial = cv2.cvtColor(trial, cv2.COLOR_RGB2GRAY)

# ret, thresh = cv2.threshold(trial, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("binarized", thresh)
# cv2.waitKey(0)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
canny = cv2.Canny(trial, 100, 200)
cv2.imshow("canny2", canny)
cv2.waitKey(0)

corners = cv2.goodFeaturesToTrack(trial, 50, 0.4, 5)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    circled = cv2.circle(canny, (x,y), 3, 255, -1)

cv2.imshow("corners", circled)
cv2.waitKey(0)
#
#
# resultingMatrix = img
#
# im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
# for i in range(0, len(contours)):
#     drawn = cv2.drawContours(resultingMatrix, contours, i, (random.randrange(255), random.randrange(255),random.randrange(255)), 3)
# cv2.imshow("contoured", drawn)
# cv2.waitKey(0)
# cnt = contours[0]
# print(cnt)
# hull = cv2.convexHull(cnt, returnPoints=False)
# defects = cv2.convexityDefects(cnt, hull)
# print(defects)
#
# for i in range(defects.shape[0]):
#     s, e, f, d = defects[i,0]
#     start = tuple(cnt[s][0])
#     end = tuple(cnt[e][0])
#     far = tuple(cnt[f][0])
#     cv2.line(canny, start, end, [255,255,255],2 )
#     cv2.circle(canny, far, 5, [255,255,255], -1)
#

# for cnt in contours:
#     print(cnt[0])
#     hull = cv2.convexHull(cnt[0], returnPoints=False)
#     defects = cv2.convexityDefects(cnt, hull)
#     print(defects)
#
#     for i in range(defects.shape[0]):
#         s, e, f, d = defects[i,0]
#         start = tuple(cnt[0][s][0])
#         end = tuple(cnt[0][e][0])
#         far = tuple(cnt[0][f][0])
#         cv2.line(canny, start, end, [0,255,0],2 )
#         cv2.circle(canny, far, 5, [0,0,255], -1)
#
# cv2.imshow("hulled", canny)
# cv2.waitKey(0)
#
#
#     print(len(cnt))

    # countLiveWorms = 0
    #
    # if len(cnt) > 200:
    #     countLiveWorms += 1

    # if len(cnt) < 100:
    #     continue

    # if not 10 < len(cnt) < 100:
    #     continue
    #
    # ellipse = cv2.fitEllipse(cnt)
    # drawn = cv2.ellipse(canny, ellipse, (255,255,0),2)
# cv2.imshow("Ellipsed", drawn)
# cv2.waitKey(0)
#     print(countLiveWorms)
#
# surf = cv2.xfeatures2d.SURF_create(15000)
# kp, des = surf.detectAndCompute(thresh, None)
# print(len(kp))
#
# img2 = cv2.drawKeypoints(thresh, kp, None, (255,0,0), 4)
# cv2.imshow("plotted", img2)
# cv2.waitKey(0)
#
# kp2, des2 = surf.detectAndCompute(sure_fg, None)
# img3 = cv2.drawKeypoints(sure_fg, kp2, None, (255,0,0), 4)
# cv2.imshow("on borders", img3)
# cv2.waitKey(0)
#
# test = cv2.imread('test_file.png')
# test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
# canny_test = cv2.Canny(test, 100, 200)
# cv2.imshow("contoured", canny_test)
# cv2.waitKey(0)
#
# minLineLength = 5000
# maxLineGap = 10
#
# try:
#     lines = cv2.HoughLinesP(canny_test, 1, np.pi/180, 30, 100, 20)
#     for x in range(0, len(lines)):
#         for x1,y1,x2,y2 in lines[x]:
#             cv2.line(canny_test,(x1,y1),(x2,y2),(255,255,0),2)
# except TypeError:
#     print("No straight lines (dead worms) found")
#
# cv2.imshow("lines", canny_test)
# cv2.waitKey(0)


for a in corners:
    for b in corners:
        # print a[0][0]
        # print b[0][1]
        if 0 < math.sqrt((a[0][0] - b[0][0])*(a[0][0] - b[0][0]) + (a[0][1] - b[0][1])*(a[0][1] - b[0][1])) < 10:
            midX = (a[0][0] + b[0][0]) / 2
            midY = (a[0][1] + b[0][1]) / 2
            cv2.circle(circled, (midX, midY), 15, (255, 255, 255))
            print (str(a[0]) + ' ' + str(b[0]) + ' show two worms intersecting')

cv2.imshow("circled", circled)
cv2.waitKey(0)