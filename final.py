import numpy as np
import cv2

img = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_A05_w2_9F329A58-2D6D-42E2-9E6D-E23ACBACE9E0.tif')
# LOAD INTO 8-BIT
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# GIVES AN APPROXIMATION OF THE ILLUMINATION EFFECT
gradient = cv2.GaussianBlur(grayscale, (55,55), 0)
# REMOVE ILLUMINATION
diff = grayscale - gradient

# TRY TO REMOVE AS MUCH NOISE AS POSSIBLE
remove = cv2.medianBlur(diff, 5)
inverse = (255 - remove)

###########################################################################
#                      IDENTIFIES NOISE BLOBS                             #
###########################################################################

# SETUP SIMPLEBLOBDETECTOR PARAMETERS
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 2
params.maxArea = 100

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByConvexity = True
params.minConvexity = 0.3

params.filterByInertia = True
params.minInertiaRatio = 0.01

# CREATE A DETECTOR WITH THE PARAMETERS
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(inverse)

# MARKS THE BLOBS THAT HAVE BEEN IDENTIFIED AS NOISE
for point in keypoints:
    point.class_id = 99

# GET ALL EDGES ON THE IMAGE
inverse = cv2.bitwise_not(inverse)
canny = cv2.Canny(inverse, 100, 200)

# CREATE MASK TO FILL IN NOISE BLOBS
canny2 = cv2.copyMakeBorder(canny, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
for point in keypoints:
    int_point = (int(point.pt[0]), int(point.pt[1]))
    cv2.floodFill(inverse, canny2, int_point, (0,0,0))

kernel = np.ones((2,2), np.uint8)

# ERODE THE REMAINING NOISE BLOBS
erosion = cv2.erode(inverse, kernel, iterations=2)
cv2.imshow("eroded", erosion)
cv2.waitKey(0)

#########################################################################

# DILATE THE REMAINING COMPONENTS TO GET THE SURE BACKGROUND
sure_bg = cv2.dilate(erosion, kernel, iterations=5)
cv2.imshow("dilated", sure_bg)
cv2.waitKey(0)