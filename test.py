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
cl2 = clahe.apply(grayscale)
cv2.imshow("CLAHE2", cl2)
cv2.waitKey(0)

cl3 = clahe.apply(gradient)
cv2.imshow("CLAHE3", cl3)
cv2.waitKey(0)

diff = grayscale - gradient
cv2.imshow("Diff", diff)
cv2.waitKey(0)

remove = cv2.medianBlur(diff, 5)
cv2.imshow("remove noise", remove)
cv2.waitKey(0)

ret, thresh = cv2.threshold(cl2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("After proccess", thresh)
cv2.waitKey(0)

# NOISE REMOVAL
kernel = np.ones((2,2), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imshow("After CLOSE", closing)
cv2.waitKey(0)

cv2.findContours(remove, contours, hierarchy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, Point(0, 0));
/// Approximate contours
vector<Rect> boundRect(contours.size());
for (unsigned int i = 0; i < contours.size(); i++)
{   //identify bounding box
    boundRect[i] = boundingRect(contours[i]);
}
for (unsigned int i = 0; i < contours.size(); i++)
{

    if ((boundRect[i].area() < //enter your area here))
    {
        src_gray(boundRect[i])=//set it to whatever value you want;
    }
}