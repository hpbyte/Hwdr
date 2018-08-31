from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from recognize.descriptor.hog import HOG
from recognize.preprocessing.deskew import Deskew
from recognize.preprocessing.center_extent import CenterExtent 
import numpy as np
import argparse
import cv2 
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "path to the dataset file")
ap.add_argument("-m", "--model", required = True,
	help = "path to where the model will be stored")
args = vars(ap.parse_args())


#load data
data = np.genfromtxt(args['dataset'], delimiter = ",", dtype = "uint8")
target = data[:, 0]
data = data[:, 1:].reshape(data.shape[0], 28, 28)

deskew = Deskew(20)
center_extent = CenterExtent((20,20))

train_data = []
preprocessors = [deskew,center_extent]

# Compute a Histogram of Oriented Gradients (HOG) by
# (optional) global image normalization
# computing the gradient image in row and col
# computing gradient histograms
# normalizing across blocks
# flattening into a feature vector

hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
	cellsPerBlock = (1, 1), transform = True)

#preprocess image
for image in data:
    for preprocessor in preprocessors:
        image = preprocessor.preprocess(image)
    
    hist = hog.describe(image)
    train_data.append(hist)


# train the model
model = LinearSVC(random_state = 42)
model.fit(train_data, target)

# dump the model to file
print("Writing model to file....")
joblib.dump(model, args["model"])

for (i,image) in enumerate(train_data[:6]):
    digit = model.predict([image])[0]
    print("Number is: {}".format(digit))
 
# cv2.imshow("Numbers",np.hstack(data[:6]))  
# cv2.imshow("DeSkew Numbers",np.hstack([deskew.preprocess(im) for im in data[:6]]))
# cv2.imshow("Center Numbers",np.hstack([center_extent.preprocess(deskew.preprocess(im)) for im in data[:6]]))
# cv2.waitKey(0)
plt.imshow(np.hstack(data[:6]))
plt.show()
plt.imshow(np.hstack([deskew.preprocess(im) for im in data[:6]]))
plt.show()
plt.imshow(np.hstack([center_extent.preprocess(deskew.preprocess(im)) for im in data[:6]]))
plt.show()
