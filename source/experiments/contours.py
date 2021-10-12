import numpy as np
import cv2 as cv
im = cv.imread('photos/photo-101.jpg', cv.IMREAD_GRAYSCALE)
#imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
imgray = cv.pyrDown(im)
imgray = cv.resize(imgray, (1440, 1080), interpolation=cv.INTER_LINEAR)

ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
imbgr = cv.merge([imgray, imgray, imgray])
cv.drawContours(imbgr, contours, -1, (0,255,0), 3)
cv.imshow('detected contours',imbgr)
cv.waitKey(0)
cv.destroyAllWindows()
