""" Extract code area
    This module takes a list of locator detections, extracts the associated rectangle and rotates it
    such that the right-angle corner is top-left and the others below it (so now its the right way up).
    To rotate an image (section) the transformation for each pixel is:
        x2 = cos(Theta)*(x1-x0) - sin(Theta)*(y1-y0) + x0
        y2 = sin(Theta)*(x1-x0) + cos(Theta)*(y1-y0) + y0
    Where x0,y0 is the centre of rotation, Theta is the rotation angle (clockwise is +ve),
    x1,y1 is the original pixel, and x2,y2 is the rotated pixel
    NB: x2,y2 are fractional so pixel intensity is an interpolation of its neighbours (see get_pixel).
"""

import const
import utils
import locator

class Extractor:

    def __init__(self, image, detections):
        self.image      = image       # grayscale image the detections were detected within (and co-ordinates apply to)
        self.detections = detections  # qualifying detections

    def get_pixel(self, x: float, y: float) -> int:
        """ get the interpolated pixel value at x,y,
            x,y are fractional so the pixel value returned is a mixture of the 4 pixels around x,y,
            the mixture is based on the ratio of the neighbours to include, the ratio of all 4 is 1,
            code based on:
                void interpolateColorPixel(double x, double y) {
                    int xL, yL;
                    xL = (int) Math.floor(x);
                    yL = (int) Math.floor(y);
                    xLyL = ipInitial.getPixel(xL, yL, xLyL);
                    xLyH = ipInitial.getPixel(xL, yL + 1, xLyH);
                    xHyL = ipInitial.getPixel(xL + 1, yL, xHyL);
                    xHyH = ipInitial.getPixel(xL + 1, yL + 1, xHyH);
                    for (int rr = 0; rr < 3; rr++) {
                        double newValue = (xL + 1 - x) * (yL + 1 - y) * xLyL[rr];
                        newValue += (x - xL) * (yL + 1 - y) * xHyL[rr];
                        newValue += (xL + 1 - x) * (y - yL) * xLyH[rr];
                        newValue += (x - xL) * (y - yL) * xHyH[rr];
                        rgbArray[rr] = (int) newValue;
                    }
                }
            from here: https://imagej.nih.gov/ij/plugins/download/Polar_Transformer.java
            explanation:
            x,y represent the top-left of a 1x1 pixel
            if x or y are not whole numbers the 1x1 pixel area overlaps its neighbours,
            the returned pixel value is the sum of the overlap fractions of its neighbour
            pixel squares, P is the fractional pixel address in its pixel, 1, 2 and 3 are
            its neighbours, dotted area is contribution from neighbours:
                +------+------+
                |  P   |   1  |
                |  ....|....  |  Ax = 1 - (Px - int(Px) = 1 - Px + int(Px) = (int(Px) + 1) - Px
                |  . A | B .  |  Ay = 1 - (Py - int(Py) = 1 - Py + int(Py) = (int(Py) + 1) - Py
                +------+------+  et al for B, C, D
                |  . D | C .  |
                |  ....|....  |
                |  3   |   2  |
                +----- +------+
            """
        cX: float = x
        cY: float = y
        xL: int = int(cX)
        yL: int = int(cY)
        xH: int = xL + 1
        yH: int = yL + 1
        pixel_xLyL = image[yL][xL]
        pixel_xLyH = image[yH][xL]
        pixel_xHyL = image[yL][xH]
        pixel_xHyH = image[yH][xH]
        if pixel_xLyL is None:
            pixel_xLyL = const.MIN_LUMINANCE
        if pixel_xLyH is None:
            pixel_xLyH = const.MIN_LUMINANCE
        if pixel_xHyL is None:
            pixel_xHyL = const.MIN_LUMINANCE
        if pixel_xHyH is None:
            pixel_xHyH = const.MIN_LUMINANCE
        ratio_xLyL = (xH - cX) * (yH - cY)
        ratio_xHyL = (cX - xL) * (yH - cY)
        ratio_xLyH = (xH - cX) * (cY - yL)
        ratio_xHyH = (cX - xL) * (cY - yL)
        part_xLyL = pixel_xLyL * ratio_xLyL
        part_xHyL = pixel_xHyL * ratio_xHyL
        part_xLyH = pixel_xLyH * ratio_xLyH
        part_xHyH = pixel_xHyH * ratio_xHyH
        pixel = int(round(part_xLyL + part_xHyL + part_xLyH + part_xHyH))
        return pixel

    def make_binary(self, detection: locator.Detection):
        """ make a binary image of the given detection """
        binary = None  # ToDo:

if __name__ == "__main__":
    """ test harness """
    import cv2

    #src = "/home/dave/precious/fellsafe/fellsafe-image/media/square-codes/square-codes-distant.jpg"
    src = "/home/dave/precious/fellsafe/fellsafe-image/source/square-codes/test-alt-bits.png"
    proximity = const.PROXIMITY_CLOSE
    #proximity = const.PROXIMITY_FAR

    logger = utils.Logger('extractor.log')
    logger.log('_test')

    located, image = locator._test(src, proximity, logger, create_new_blobs=True)

    extractor = Extractor(image, located.detections())
