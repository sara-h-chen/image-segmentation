import numpy as np
import cv2
from matplotlib import pyplot as plt

# LOAD IMAGE
img = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_A05_w2_9F329A58-2D6D-42E2-9E6D-E23ACBACE9E0.tif')
# print (len(img.shape))
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gradient = cv2.GaussianBlur(grayscale, (55,55), 0)
cv2.imshow("gradient", gradient)
cv2.waitKey(0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(60, 60))
# cl1 = clahe.apply(gradient)
# cv2.imshow("After CLAHE", cl1)
# cv2.waitKey(0)

# ret, thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow("filtered", thresh)
# cv2.waitKey(0)

# equ = cv2.equalizeHist(grayscale)
# cv2.imshow("Equalized", equ)
# cv2.waitKey(0)
# cl2 = clahe.apply(grayscale)
# cv2.imshow("CLAHE2", cl2)
# cv2.waitKey(0)

diff = grayscale - gradient
cv2.imshow("Diff", diff)
cv2.waitKey(0)

remove = cv2.medianBlur(diff, 5)
cv2.imshow("remove noise", remove)
cv2.waitKey(0)

# NOISE REMOVAL
# kernel = np.ones((2,2), np.uint8)
# closing = cv2.morphologyEx(remove, cv2.MORPH_CLOSE, kernel, iterations=2)
# cv2.imshow("After CLOSE", closing)
# cv2.waitKey(0)

inverse = (255 - remove)
#
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 2
params.maxArea = 100

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(inverse)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(inverse, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

for point in keypoints:
    point.class_id = 99
    print (point.pt)
    print (point.class_id)