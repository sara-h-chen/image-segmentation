import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_A05_w2_9F329A58-2D6D-42E2-9E6D-E23ACBACE9E0.tif')
# LOAD INTO 8-BIT
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# GIVES AN APPROXIMATION OF THE ILLUMINATION
gradient = cv2.GaussianBlur(grayscale, (55,55), 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(60, 60))