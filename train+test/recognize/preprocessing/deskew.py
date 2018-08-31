import imutils 
import numpy as np
import cv2

class Deskew:
    def __init__(self,width):
        self.width = width
    
    def preprocess(self,image):
        # grab the width and height of the image and compute
        # moments for the image
        (h, w) = image.shape[:2]
        moments = cv2.moments(image)

        # deskew the image by applying an affine transformation
        skew = moments["mu11"] / moments["mu02"]
        M = np.float32([[1, skew, -0.5 * w * skew],[0, 1, 0]])
        image = cv2.warpAffine(image, M, (w, h),
        flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        # resize the image to have a constant width
        image = imutils.resize(image, width = self.width)

        # return the deskewed image
        return image