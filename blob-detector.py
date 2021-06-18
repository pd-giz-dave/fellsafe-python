# Standard imports
import cv2
import numpy as np;

# Read image
orig = cv2.imread("101-341-511-682-795-877-984-photo.jpg", cv2.IMREAD_GRAYSCALE)

# De-noise it
blurred = cv2.GaussianBlur(orig, (3, 3), 1)

# Downsize it
width, height = blurred.shape
aspect_ratio = width / height
new_width = 1024
new_height = int(new_width * aspect_ratio)
shrunk = cv2.resize(blurred, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Reverse it
im = 255-shrunk

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 256
params.thresholdStep = 8

# Filter by Area.
params.filterByArea = True
params.minArea = 100
params.maxArea = 100000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs (NB: it looks for dark connected areas - hence earlier image reversal)
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
