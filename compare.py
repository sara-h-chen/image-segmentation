import numpy as np
import cv2
import os

# GETS THE FOLDER WITH ALL GROUND TRUTH
dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ground_truth'))
onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
print(onlyfiles)
cv2.imread("A05_binary.png")


# DEDUCT THE GROUND TRUTH IMAGE FROM SURE_FG. THIS IDENTIFIES NOISE BLOBS THAT YOU HAVE MISSED DURING PROCESSING.
# DEDUCT SURE_BG FROM GROUND TRUTH IMAGE. THIS IDENTIFIES WORMS YOU MAY HAVE MISSED.
# ADD THE TWO TOGETHER AND COUNT THE NUMBER cv2.countNonZero(src), DIVIDE BY img.size AND GET THE ERROR PERCENTAGE

# COUNT THE LENGTH OF THE NUMBER OF FILES AND THEN DEDUCT THAT NUMBER FROM THE NUMBER OF WORMS YOU DETECTED