import cv2

img_file = 'debug_images/_photo-many-v1_315x402y-flat.png'
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 10, 20)
cv2.imshow('title', edges)
cv2.waitKey(0)
