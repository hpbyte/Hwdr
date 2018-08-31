import imutils
import mahotas
import numpy as np 
import cv2

class CenterExtent:
    def __init__(self,size):
        self.size = size 
    

    def preprocess(self,image):
        # grab the extent width and height
        (eW, eH) = self.size

        # handle when the width is greater than the height
        if image.shape[1] > image.shape[0]:
            image = imutils.resize(image, width = eW)

        # otherwise, the height is greater than the width
        else:
            image = imutils.resize(image, height = eH)

        # allocate memory for the extent of the image and
        # grab it
        extent = np.zeros((eH, eW), dtype = "uint8")
        offsetX = (eW - image.shape[1]) // 2
        offsetY = (eH - image.shape[0]) // 2
        extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image

        # compute the center of mass of the image and then
        # move the center of mass to the center of the image
        (cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")
        (dX, dY) = ((self.size[0] // 2) - cX, (self.size[1] // 2) - cY)
        M = np.float32([[1, 0, dX], [0, 1, dY]])
        extent = cv2.warpAffine(extent, M, self.size)

        # return the extent of the image
        return extent