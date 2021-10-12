import cv2
from matplotlib import pyplot as plt

img = cv2.imread('15-segment-angle-test-photo.jpg',cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('orig',cv2.WINDOW_NORMAL)
cv2.imshow('orig',img)
if False:
    blurred = cv2.bilateralFilter(img,7)
    cv2.namedWindow('blurred',cv2.WINDOW_NORMAL)
    cv2.imshow('blurred',blurred)
    binary = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.namedWindow('binary',cv2.WINDOW_NORMAL)
    cv2.imshow('binary',binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
