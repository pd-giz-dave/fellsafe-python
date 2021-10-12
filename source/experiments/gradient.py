import cv2

image = cv2.imread('/home/dave/precious/fellsafe/fellsafe-image/debug_images/_photo-many-v1/_photo-many-v1-11x416y/_photo-many-v1_11x416y-04-flat.png')
cv2.imshow("Image", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
b2wX = cv2.Sobel(gray, ddepth=-1, dx=1, dy=0, ksize=3)
w2bX = cv2.Sobel(255-gray, ddepth=-1, dx=1, dy=0, ksize=3)
b2wY = cv2.Sobel(gray, ddepth=-1, dx=0, dy=1, ksize=3)
w2bY = cv2.Sobel(255-gray, ddepth=-1, dx=0, dy=1, ksize=3)
# subtract the y-gradient from the x-gradient
gradX = cv2.add(b2wX, w2bX)
gradY = cv2.add(b2wY, w2bY)
gradXY = cv2.subtract(gradX, gradY)

cv2.imshow("GradX", gradX)
cv2.imshow("GradY", gradY)
cv2.imshow("GradXY", gradXY)
cv2.waitKey(0)
