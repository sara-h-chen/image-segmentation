############################################################
#     ADAPTED FROM:                                        #
#     http://stackoverflow.com/questions/12995434/         #
#     representing-and-solving-a-maze-given-an-image       #
############################################################

import sys
import cv2
import numpy as np

class Colourer(object):

    set = 1
    width = 0
    height = 0

    def __init__(self, imgwidth, imgheight):
        self.width = imgwidth
        self.height = imgheight

    def isWhite(self, value):
        tuplefied = (value[0], value[1], value[2])
        if tuplefied == (255, 255, 255):
            return True

    def getAdjacent(self, n):
        x,y = n
        return [(max(x-1,0),y),(x,max(0,y-1)),(min(x+1,self.width),y),(x,min(y+1,self.height))]

    def BFS(self, pixel, coordinatePair):
        for adjacent in self.getAdjacent(coordinatePair):
            x,y = adjacent
            # print adjacent[0]
            # print adjacent[1]
            # break
            if self.isWhite(image[x][y]):
                pixel[0] = 127
                pixel[1] = 127
                pixel[2] = 127
    #             "color" each pixel with a class_id of set
        self.set += 1

if __name__ == '__main__':

    image = cv2.imread('test_file.png')
    clrer = Colourer(np.size(image, 1), np.size(image, 0))
    startingPoints = []

    for i in range(0, clrer.height):
        for j in range(0, clrer.width):
            coordinatePair = (i,j)
            # print(coordinatePair)
            pixelTuple = (image[i][j][0], image[i][j][1], image[i][j][2])
            # print pixelTuple
            if clrer.isWhite(pixelTuple):
                startingPoints.append((i,j))
                clrer.BFS(image[i][j], coordinatePair)
                break

    print(startingPoints)
    print(clrer.set)