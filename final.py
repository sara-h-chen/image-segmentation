#################################################################################
#  ALL GROUND TRUTH IMAGES OF INDIVIDUAL WORMS SHOULD BE PLACED INTO THE        #
#  ground_truth FOLDER SO THAT COMPARISONS CAN BE MADE. ALL IMAGES USED TO      #
#  PRODUCE/TEST THIS ALGORITHM IS SUPPLIED IN THE SAME FOLDER. THE ALGORITHM    #
#  ACCEPTS FILES SUBMITTED TO IT AS PARAMETERS BUT MAY DETECT WORMS, COUNT AND  #
#  CLASSIFY THEM WITH LESS ACCURACY.                                            #
#                                                                               #
#  Accepts files as arguments. Reads and processes the image by removing the    #
#  background, then detecting the remaining contours to identify worms. Worms   #
#  are labelled and counted. Information on clusters are extracted from the     #
#  corners in the image. Dead worms are annotated, based on their texture.      #
# ----------------------------------------------------------------------------- #
# REFERENCES: http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html    #
#             https://www.learnopencv.com/blob-detection-using-opencv-python-c/ #
#             http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/    #
#             py_shi_tomasi/py_shi_tomasi.html                                  #
#################################################################################

import numpy as np
import math
import random
import argparse
import re
import cv2


##############################################################################
#              PROCESSES IMAGES WITH ILLUMINATED BACKGROUND                  #
##############################################################################

def processIlluminatedBg(grayscale):

    # GIVES AN APPROXIMATION OF THE ILLUMINATION EFFECT
    gradient = cv2.GaussianBlur(grayscale, (55, 55), 0)
    # REMOVE ILLUMINATION
    diff = grayscale - gradient

    # TRY TO REMOVE AS MUCH NOISE AS POSSIBLE
    remove = cv2.medianBlur(diff, 5)
    inverse = (255 - remove)

    return removeNoiseBlobs(inverse)


##############################################################################
#                   PROCESSES IMAGES WITH DARK BACKGROUND                    #
##############################################################################

def processDarkBg(grayscale):

    # INCREASES CONTRAST ON IMAGE WITH HISTOGRAM EQUALIZATION
    equalize = cv2.equalizeHist(grayscale)

    return removeNoiseBlobs(equalize)


##############################################################################
#            BINARY THRESHOLDING FOR IMAGES WITH DARK BACKGROUND             #
##############################################################################

def binaryOtsu(backgrounds):

    sure_fg = backgrounds[0]

    ret, thresh = cv2.threshold(sure_fg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # APPLY CLOSE TO REDUCE NOISE
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    return thresh


##############################################################################
#                         IDENTIFIES NOISE BLOBS                             #
##############################################################################

def removeNoiseBlobs(inverse):

    # SETUP SIMPLEBLOBDETECTOR PARAMETERS
    params = cv2.SimpleBlobDetector_Params()

    # FILTER BY AREA
    params.filterByArea = True
    params.minArea = 2
    params.maxArea = 150

    # FILTER BY CIRCULARITY
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # FILTER BY CONVEXITY
    params.filterByConvexity = True
    params.minConvexity = 0.3

    # FILTER BY INERTIA
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # CREATE A DETECTOR WITH THE PARAMETERS TO DETECT NOISE BLOBS
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        noiseDetector = cv2.SimpleBlobDetector(params)
    else:
        noiseDetector = cv2.SimpleBlobDetector_create(params)

    noiseKeypoints = noiseDetector.detect(inverse)

    # MARKS THE BLOBS THAT HAVE BEEN IDENTIFIED AS NOISE
    for point in noiseKeypoints:
        point.class_id = 99

    # GET ALL EDGES ON THE IMAGE
    inverse = cv2.bitwise_not(inverse)
    canny = cv2.Canny(inverse, 100, 200)

    # CREATE MASK TO FILL IN NOISE BLOBS
    canny2 = cv2.copyMakeBorder(canny, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    for point in noiseKeypoints:
        int_point = (int(point.pt[0]), int(point.pt[1]))
        cv2.floodFill(inverse, canny2, int_point, (0, 0, 0))

    kernel = np.ones((2, 2), np.uint8)

    # ERODE THE REMAINING NOISE BLOBS AND GET THE CONFIRMED FOREGROUND
    sure_fg = cv2.erode(inverse, kernel, iterations=2)

    if args.verbose:
        cv2.imshow("SURE FOREGROUND", sure_fg)
        cv2.waitKey(0)

    # DILATE THE REMAINING SHAPES TO FACILITATE SHAPE DETECTION
    sure_bg = cv2.dilate(sure_fg, kernel, iterations=3)

    if args.verbose:
        cv2.imshow("SURE BACKGROUND", sure_bg)
        cv2.waitKey(0)

    return (sure_fg, sure_bg)


##############################################################################
#               GIVE SEPARATE COMPONENTS DISTINCT COLOURS                    #
##############################################################################

def segmentWithColors(backgrounds):

    # IF DARK BACKGROUND, APPLY OTSU'S BINARIZATION
    if len(backgrounds) == 2:
        clearEdges = cv2.Canny(backgrounds[1], 100, 200)
        if args.verbose:
            cv2.imshow("EDGES AFTER CANNY", clearEdges)
            cv2.waitKey(0)
    else:
        clearEdges = cv2.Canny(backgrounds, 100, 200)
        if args.verbose:
            cv2.imshow("OTSU BINARIZATION APPLIED", clearEdges)
            cv2.waitKey(0)

    # TURN IMAGE BACK INTO COLOR IMAGE
    coloredComponents = cv2.cvtColor(clearEdges, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(clearEdges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(contours)):
        cv2.drawContours(coloredComponents, contours, i, (random.randrange(255), random.randrange(255), random.randrange(255)), 3)

    cv2.imshow("SEGMENTATION WITH COLORED CONTOURS", coloredComponents)
    cv2.waitKey(0)


##############################################################################
#                    DETECT WORMS TO GET THEIR KEYPOINTS                     #
##############################################################################

def getKeypoints(backgrounds):

    sure_fg = backgrounds[0]
    sure_bg = backgrounds[1]

    # INVERSE AGAIN TO DETECT CONVEXITY
    inverseForConvexity = cv2.bitwise_not(sure_bg)

    # SETUP SIMPLEBLOBDETECTOR WITH PARAMETERS
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 9999

    params.filterByConvexity = True
    params.minConvexity = 0.01

    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # CREATE A DETECTOR WITH THE PARAMETERS
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        wormDetector = cv2.SimpleBlobDetector(params)
    else:
        wormDetector = cv2.SimpleBlobDetector_create(params)

    # DETECT SHAPES
    keypointsOnLight = wormDetector.detect(inverseForConvexity)
    totalKeypoints = wormDetector.detect(sure_bg)

    # DRAW ON DETECTED SHAPED WITH RED CIRCLES
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ENSURES THE SIZE OF THE CIRCLE CORRESPONDS TO THE SIZE OF BLOB
    dark_on_light_keypoints = cv2.drawKeypoints(sure_fg, keypointsOnLight, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    light_on_dark_keypoints = cv2.drawKeypoints(inverseForConvexity, totalKeypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # UNCOMMENT HERE TO SHOW KEYPOINTS
    cv2.imshow("WORMS DETECTED USING KEYPOINTS", dark_on_light_keypoints)
    cv2.waitKey(0)
    cv2.imshow("IMAGE INVERSED TO FILTER FOR CONVEXITY", light_on_dark_keypoints)
    cv2.waitKey(0)

    # MERGE THE TWO KEYPOINT SETS
    for key in keypointsOnLight:
        if key not in totalKeypoints:
            totalKeypoints.append(key)

    # LABEL EACH KEYPOINT AS DISTINCT OBJECT
    for i in range(0, len(totalKeypoints)):
        totalKeypoints[i].class_id = i + 1

    return totalKeypoints


##############################################################################
#         IDENTIFY OVERLAP/INTERSECTION BETWEEN CLUSTERED WORMS             #
##############################################################################

def identifyCluster(binarized):

    # GET EDGES TO ANNOTATE ON
    canny = cv2.Canny(binarized, 100, 200)

    # APPLY SHI-TOMASI ALGORITHM TO FIND CORNERS
    corners = cv2.goodFeaturesToTrack(binarized, 50, 0.4, 5)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(canny, (x, y), 3, 255, -1)

    if args.verbose:
        cv2.imshow("IDENTIFIED CORNERS", canny)
        cv2.waitKey(0)

    # GET DISTANCE BETWEEN IDENTIFIED CORNERS
    # IF CORNERS ARE CLOSE, THEY ARE LIKELY TO BE INTERSECTION POINTS
    for a in corners:
        for b in corners:
            if 0 < math.sqrt(((a[0][0] - b[0][0]) * (a[0][0] - b[0][0])) + ((a[0][1] - b[0][1]) * (a[0][1] - b[0][1]))) < 10:
                midpointX = (a[0][0] + b[0][0]) / 2
                midpointY = (a[0][1] + b[0][1]) / 2
                cv2.circle(canny, (midpointX, midpointY), 15, (255,255,255))
                print ("CLOSE CORNERS: " + str(a[0]) + " & " + str(b[0]) + " suggests two worms are intersecting")

    cv2.imshow("INTERSECTIONS DETECTED", canny)
    cv2.waitKey(0)


##############################################################################
#              IDENTIFY DEAD WORMS WITH DISCONTINUOUS BORDERS                #
##############################################################################

def detectDead(binarized):

    canny = cv2.Canny(binarized, 100, 200)
    # IDENTIFY ALL BORDERS IN THE CANNY IMAGE
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        # IGNORE WORMS WITH CONTINUOUS BORDERS & NOISE BLOBS
        if not 10 < len(cnt) < 35:
            continue

        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(canny, ellipse, (255,255,255), 2)

    cv2.imshow("ANNOTATED DEAD WORMS", canny)
    cv2.waitKey(0)

    return canny

############################################################################
#                             MAIN METHOD                                  #
############################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Processes the image supplied to detect worms. "
                                                 "WARNING: Naming convention of file based on image channel must remain consistent, with _w1_ indicating images with a dark background (no background) and _w2_ indicating images with an illuminated background.")
    parser.add_argument("file", help=("The raw file for processing"))
    # parser.add_argument("groundtruth", help=("The ground truth image with all the worms in binary"))
    parser.add_argument("-v", "--verbose", help=("Displays all steps in the process; increases output verbosity"),
                        action="store_true")

    # RETURNS ARGUMENT AS args.file
    args = parser.parse_args()

    # CHECK FOR IMAGE CHANNEL
    w1 = re.compile(r'\ww1\w')
    w2 = re.compile(r'\ww2\w')

    if w1.search(args.file):
        imgdark = cv2.imread(args.file)

        grey = cv2.cvtColor(imgdark, cv2.COLOR_BGR2GRAY)
        darkbackground = processDarkBg(grey)
        segmentWithColors(binaryOtsu(darkbackground))
        print("Worms found in '" + args.file + "': " + str(len(getKeypoints(darkbackground))))
        identifyCluster(binaryOtsu(darkbackground))
        try:
            detectDead(binaryOtsu(darkbackground))
        except UnboundLocalError:
            print("No dead worms found")

    elif w2.search(args.file):
        img = cv2.imread(args.file)

        # LOAD INTO BINARY
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        separatedbackgrounds = processIlluminatedBg(grayscale)
        segmentWithColors(separatedbackgrounds)

        print ("Worms found in '" + args.file + "': " + str(len(getKeypoints(separatedbackgrounds))))
        # TAKE ONLY THE SURE BACKGROUND
        identifyCluster(separatedbackgrounds[1])
        try:
            # TAKE ONLY THE SURE FOREGROUND
            detectDead(separatedbackgrounds[0])
        except UnboundLocalError:
            print("No dead worms found")

    else:
        print("FAILED: Naming convention of file not followed.")
