#################################################################################
#  # TODO: Fill up documentation here
#
# ----------------------------------------------------------------------------- #
# REFERENCES: http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html    #
#             https://www.learnopencv.com/blob-detection-using-opencv-python-c/ #
#################################################################################

import numpy as np
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
    cv2.imshow("sure_fg", sure_fg)
    cv2.waitKey(0)

    # DILATE THE REMAINING SHAPES TO FACILITATE SHAPE DETECTION
    sure_bg = cv2.dilate(sure_fg, kernel, iterations=3)
    cv2.imshow("sure_bg", sure_bg)
    cv2.waitKey(0)

    return (sure_fg, sure_bg)


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
    cv2.imshow("Keypoints", dark_on_light_keypoints)
    cv2.waitKey(0)
    cv2.imshow("Keypoints", light_on_dark_keypoints)
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
#        APPLY CANNY TO IDENTIFY WORMS WITH DISCONTINUOUS BORDERS            #
##############################################################################

def detectDead(binarized):

    canny = cv2.Canny(binarized, 100, 200)
    # IDENTIFY ALL BORDERS IN THE CANNY IMAGE
    im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        # IGNORE WORMS WITH CONTINUOUS BORDERS & NOISE BLOBS
        if not 10 < len(cnt) < 35:
            continue

        ellipse = cv2.fitEllipse(cnt)
        discontinuousPatches = cv2.ellipse(canny, ellipse, (255,255,255), 2)

    cv2.imshow("results", discontinuousPatches)
    cv2.waitKey(0)

    return discontinuousPatches

############################################################################
#                             MAIN METHOD                                  #
############################################################################

if __name__ == '__main__':

    # TODO: Parse the name of the file to decide which route to take
    img = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_A05_w2_9F329A58-2D6D-42E2-9E6D-E23ACBACE9E0.tif')

    # LOAD INTO BINARY
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    separatedbackgrounds = processIlluminatedBg(grayscale)
    # TODO: Get the filename and print it here together
    print ("Worms found: " + str(len(getKeypoints(separatedbackgrounds))))
    try:
        # TAKE ONLY THE SURE FOREGROUND
        detectDead(separatedbackgrounds[0])
    except UnboundLocalError:
        print("No dead worms found")


    imgdark = cv2.imread('1649_1109_0003_Amp5-1_B_20070424_A01_w1_9E84F49F-1B25-4E7E-8040-D1BB2D7E73EA.tif')

    grey = cv2.cvtColor(imgdark, cv2.COLOR_BGR2GRAY)
    darkbackground = processDarkBg(grey)
    print("Worms found: " + str(len(getKeypoints(darkbackground))))
    try:
        detectDead(binaryOtsu(darkbackground))
    except UnboundLocalError:
        print("No dead worms found")
