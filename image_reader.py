##############################################################################
#  REFERENCE: http://www.bogotobogo.com/python/OpenCV_Python                 #
# /python_opencv3_Image_Watershed_Algorithm_Marker_Based_Segmentation.php    #
##############################################################################
#  REFERENCE: http://docs.opencv.org/trunk/d7/d4d/                           #
#  tutorial_py_thresholding.html                                             #
##############################################################################

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# LOAD IMAGE
img = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_A05_w2_9F329A58-2D6D-42E2-9E6D-E23ACBACE9E0.tif')


# THRESHOLDING MEANS THAT A PIXEL IS EITHER WHITE OR BLACK, AFTER A CERTAIN VALUE
# OTSU BINARIZATION CALCULATES THE THRESHOLD VALUE FROM THE IMAGE HISTOGRAM
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

equ = cv2.equalizeHist(grayscale)
cv2.imshow("Equalized", equ)
cv2.waitKey(0)
# ret2, thresh2 = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# closing2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=2)
# cv2.imshow("After contrast", closing2)
# cv2.waitKey(0)

inverse = (255 - equ)
cv2.imshow("inverse", inverse)
cv2.waitKey(0)

cv2.imshow("Otsu's+Thresh", thresh)
cv2.waitKey(0)

# NOISE REMOVAL
kernel = np.ones((2,2), np.uint8)
closing = cv2.morphologyEx(equ, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imshow("close morph", closing)
cv2.waitKey(0)

# inverse = (255 - closing)
# cv2.imshow("inverse", inverse)
# cv2.waitKey(0)

# plt.subplot(131), plt.imshow(grayscale)
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(thresh, 'gray')
# plt.title("Otsu's Binary Threshold + Gaussian Blur"), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(closing, 'gray')
# plt.title("morphologyEx"), plt.xticks([]), plt.yticks([])
# plt.show()
