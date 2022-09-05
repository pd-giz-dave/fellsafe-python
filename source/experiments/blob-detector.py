# Standard imports
import cv2
import numpy as np

# Read image
orig = cv2.imread("/home/dave/precious/fellsafe/fellsafe-image/media/photo-332-222-555-800-574-371-757-611-620-132-mid.jpg", cv2.IMREAD_GRAYSCALE)

# Downsize it
width, height = orig.shape
aspect_ratio = width / height
new_width = 2048
new_height = int(new_width * aspect_ratio)
shrunk = cv2.resize(orig, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# De-noise it
blurred = shrunk #cv2.medianBlur(shrunk, 3)

# Reverse it - not if filter by colour == 255
#im = 255-blurred
im = blurred

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0 + 32
params.maxThreshold = 256 - 32
params.thresholdStep = 32

# Filter by Area.
params.filterByArea = True
params.minArea = 10
params.maxArea = 1000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.7

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.8

# Filter by Colour
params.filterByColor = True
params.blobColor = 255

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs (NB: it looks for dark connected areas - hence earlier image reversal)
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(shrunk, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imwrite('blobs.jpg', im_with_keypoints)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
