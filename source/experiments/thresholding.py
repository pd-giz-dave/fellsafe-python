# import the necessary packages
import argparse
import cv2
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
#args = vars(ap.parse_args())

# load the image and display it
#image = cv2.imread(args["image"])
image = cv2.imread('/home/dave/precious/fellsafe/fellsafe-image/debug_images/_photo-many-v1/_photo-many-v1-762x1016y/_photo-many-v1_762x1016y-04-flat.png')
cv2.imshow("Image", image)
# convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# apply simple thresholding with a hardcoded threshold value
#(T, threshInv) = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("Simple Thresholding", threshInv)
#cv2.waitKey(0)

# apply Otsu's automatic thresholding
#(T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#cv2.imshow("Otsu Thresholding", threshInv)
#cv2.waitKey(0)

# instead of manually specifying the threshold value, we can use
# adaptive thresholding to examine neighborhoods of pixels and
# adaptively threshold each neighborhood
#thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
#cv2.imshow("Mean Adaptive Thresholding", thresh)
#cv2.waitKey(0)

# perform adaptive thresholding again, this time using a Gaussian
# weighting versus a simple mean to compute our local threshold
# value
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 0)
cv2.imshow("Gaussian Adaptive Thresholding", thresh)
cv2.waitKey(0)

kernelSizes = [(3, 3), (5, 5), (7, 7)]
for kernelSize in kernelSizes:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	cv2.imshow("Opening: ({}, {})".format(kernelSize[0], kernelSize[1]), opening)
	cv2.waitKey(0)
