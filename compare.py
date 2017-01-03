#################################################################################
#  ALL GROUND TRUTH IMAGES OF INDIVIDUAL WORMS SHOULD BE PLACED IN THE          #
#  ground_truth FOLDER. THE BINARY IMAGE OF THE GROUND TRUTH IMAGE CONTAINING   #
#  ALL THE WORMS IN A SINGLE PICTURE SHOULD BE PASSED INTO THIS SCRIPT AS AN    #
#  ARGUMENT ON THE COMMAND LINE.                                                #
#                                                                               #
#  Compares the results from the worm detection algorithm to the ground truth   #
#  images supplied. Runs separately from the main algorithm, i.e. comparisons   #
#  will be made only when this script is run.                                   #
#################################################################################

import numpy as np
import final
import cv2
import os


##############################################################################
#        FINDS ACCURACY OF NOISE REMOVAL AND BACKGROUND SEPARATION           #
##############################################################################

# def compareIndividual():

# GETS THE FOLDER WITH ALL INDIVIDUAL GROUND TRUTH
dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ground_truth'))
onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
print(onlyfiles)
print(len(onlyfiles))

# USE THIS IN ARGPARSE AND MAKE IT COMPULSORY
img = cv2.imread("A05_binary.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sure_bg = cv2.imread("sure_bg.png")
sure_bg = cv2.cvtColor(sure_bg, cv2.COLOR_BGR2GRAY)

# GIVES ABSOLUTE DIFFERENCE; REGIONS OF SOLID WHITE INDICATE DIFFERENCES.
deduct = cv2.absdiff(sure_bg, img)
cv2.imshow("deduct sure_bg", deduct)
cv2.waitKey(0)

# ADD THE TWO TOGETHER AND COUNT THE NUMBER cv2.countNonZero(src), DIVIDE BY img.size AND GET THE ERROR PERCENTAGE
size = img.size
print(size)
colored = cv2.countNonZero(img)
print(colored)

error_rate = (colored / float(size)) * 100
print ("Error rate: {0:.2f}%".format(error_rate))

# COUNT THE LENGTH OF THE NUMBER OF FILES AND THEN DEDUCT THAT NUMBER FROM THE NUMBER OF WORMS YOU DETECTED