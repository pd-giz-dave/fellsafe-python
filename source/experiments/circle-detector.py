
import cv2
import numpy as np

# Read image.
img = cv2.imread('photos/photo-365-oblique.jpg', cv2.IMREAD_COLOR)

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
blurred = cv2.blur(gray, (3, 3))

# Downsize it
width, height = blurred.shape
aspect_ratio = width / height
new_width = 1024
new_height = int(new_width * aspect_ratio)
shrunk = cv2.resize(blurred, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(shrunk,
                                    cv2.HOUGH_GRADIENT, 1, 10,
                                    param1=120, param2=30,
                                    minRadius=10, maxRadius=80)

# Draw circles that are detected.
if detected_circles is not None:

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(shrunk, (a, b), r, (0, 255, 0), 2)

        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(shrunk, (a, b), 1, (0, 0, 255), 3)

    cv2.imwrite('circles.jpg', shrunk)
    cv2.imshow("Detected Circle", shrunk)
    cv2.waitKey(0)
