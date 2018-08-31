from sklearn.externals import joblib
from recognize.descriptor.hog import HOG
from recognize.preprocessing.deskew import Deskew
from recognize.preprocessing.center_extent import CenterExtent 
import numpy as np
import imutils
import mahotas
import argparse
import cv2 
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model")
args = vars(ap.parse_args())

model = joblib.load(args["model"])

hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), transform = True)
deskew = Deskew(20)
center_extent = CenterExtent((20,20))
preprocessors = [deskew,center_extent]


image = cv2.imread(args["image"])
image = imutils.resize(image,width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])


for (c,_) in cnts:
	(x,y,w,h) = cv2.boundingRect(c)

	roi = gray[y:y+h,x:x+w]
	thresh = roi.copy()
	# (T,thresh) = cv2.threshold(thresh,150,255,cv2.THRESH_BINARY_INV)
	T = mahotas.thresholding.otsu(roi)
	thresh[thresh > T] = 255
	thresh = cv2.bitwise_not(thresh)
	for preprocessor in preprocessors:
		thresh = preprocessor.preprocess(thresh)
	hist = hog.describe(thresh)
	hist = hist.reshape(1,-1)
	digit = model.predict(hist)[0]

	cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
	cv2.putText(image,str(digit),(x-8,y-8),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
	# cv2.imshow("image",image)
	# cv2.waitKey(0)
	plt.imshow(image)
	plt.show()
cv2.imwrite("output.png",image)