import numpy as np
import cv2

img = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_A05_w2_9F329A58-2D6D-42E2-9E6D-E23ACBACE9E0.tif')
# Load into 8-bit
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Gives an approximation of the illumination effect
gradient = cv2.GaussianBlur(grayscale, (55,55), 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(60, 60))

diff = grayscale - gradient

# Try to remove as much noise as possible
remove = cv2.medianBlur(diff, 5)

inverse = (255 - remove)

###########################################################################
#                      IDENTIFIES NOISE BLOBS                             #
###########################################################################

# Setup SimpleBlobDetector parameters
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

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(inverse)

# Draw detected blobs as red circles
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(inverse, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

# Marks the blobs that have been identified as noise
for point in keypoints:
    point.class_id = 99
