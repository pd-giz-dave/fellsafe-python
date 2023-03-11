""" wrapper around cv2 to hide it from everyone
    these functions manipulate a 2D array of luminance values,
    it uses the opencv library to read/write images and numpy arrays to modify images at the pixel level
    """

import numpy as np
import cv2
import os
import pathlib
import const

""" WARNING
    cv2 and numpy's co-ordinates are backwards from our pov, the 'x' co-ordinate are columns and 'y' rows.
    The functions here use 'x' then 'y' parameters, swapping as required when dealing with cv2 and numpy arrays.
    NB: The final implementation will not use cv2 (its too big) so its use here is purely a diagnostic convenience.
        Significant algorithms are implemented DIY. 
    """

def copy(buffer):
    """ return a copy of the given buffer """
    return np.copy(buffer)

def new(width, height, luminance=const.MIN_LUMINANCE):
    """ prepare a new buffer of the given size and luminance """
    return np.full((height, width), luminance, dtype=np.uint8)  # NB: numpy arrays follow cv2 conventions

def size(buffer):
    """ return the x,y size of the given buffer """
    max_x = buffer.shape[1]  # NB: cv2 x, y are reversed
    max_y = buffer.shape[0]  # ..
    return max_x, max_y

def load(image_file):
    """ load a buffer from an image file as a grey scale image and return it """
    return cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

def unload(buffer, image_file):
    """ unload the given buffer to a PNG image file """
    cv2.imwrite(image_file, buffer)

def downsize(buffer, new_size):
    """ downsize the image such that either its width or height is at most that given,
        the aspect ratio is preserved, its a no-op if image already small enough,
        returns a new buffer of the new size,
        this is purely a diagnostic aid to simulate low-resolution cameras
        """
    width, height = size(buffer)
    if width <= new_size or height <= new_size:
        # its already small enough
        return buffer
    if width > height:
        # bring height down to new size
        new_height = new_size
        new_width = int(width / (height / new_size))
    else:
        # bring width down to new size
        new_width = new_size
        new_height = int(height / (width / new_size))
    return cv2.resize(buffer, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

def upsize(buffer, scale: float):
    """ return a buffer that is scale times bigger in width and height than that given,
        the contents of the given buffer are also scaled by interpolating neighbours (using makepixel),
        scale must be positive and greater then 1,
        each source pixel is considered to consist of scale * scale sub-pixels, each destination pixel
        is one sub-pixel and is constructed from an interpolation of a 1x1 source pixel area centred on
        the sub-pixel
        """
    if scale <= 1:
        return buffer
    max_x, max_y = size(buffer)
    width  = int(round(max_x * scale))
    height = int(round(max_y * scale))
    upsized = new(width, height)
    for dest_x in range(width):
        src_x = dest_x / scale
        for dest_y in range(height):
            src_y = dest_y / scale
            pixel = makepixel(buffer, src_x, src_y)
            upsized[dest_y, dest_x] = pixel
    return upsized

def show(buffer, title='title'):
    """ show the given buffer """
    cv2.imshow(title, buffer)
    cv2.waitKey(0)

def getpixel(buffer, x, y):
    """ get the pixel value at x,y
        nb: cv2 x,y is reversed from our pov
        """
    max_x, max_y = size(buffer)
    if x < 0 or x >= max_x or y < 0 or y >= max_y:
        # out of range pixel
        return None
    else:
        return buffer[y, x]  # NB: cv2 x, y are reversed

def putpixel(buffer, x, y, value):
    """ put the pixel of value at x,y
        value may be a greyscale value or a colour tuple
        """
    max_x, max_y = size(buffer)
    if x < 0 or x >= max_x or y < 0 or y >= max_y:
        return
    buffer[y, x] = value  # NB: cv2 x, y are reversed

def pixelparts(buffer, x: float, y: float) -> ():
    """ get the neighbour parts contributions for a pixel at x,y
        x,y are fractional so the pixel contributions is a mixture of the 4 pixels around x,y,
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
        the pixel value is the sum of the overlap fractions of its neighbour pixel squares,
        P is the fractional pixel address in its pixel, 1, 2 and 3 are its neighbours,
        dotted area is contribution from neighbours:
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
    max_x, max_y = size(buffer)
    cX: float = x
    cY: float = y
    xL: int = int(cX)
    yL: int = int(cY)
    xH: int = xL + 1
    yH: int = yL + 1
    xU = min(xH, max_x - 1)
    yU = min(yH, max_y - 1)
    pixel_xLyL = buffer[yL][xL]
    pixel_xLyH = buffer[yU][xL]
    pixel_xHyL = buffer[yL][xU]
    pixel_xHyH = buffer[yU][xU]
    ratio_xLyL = (xH - cX) * (yH - cY)
    ratio_xHyL = (cX - xL) * (yH - cY)
    ratio_xLyH = (xH - cX) * (cY - yL)
    ratio_xHyH = (cX - xL) * (cY - yL)
    return (pixel_xLyL, ratio_xLyL), (pixel_xHyL, ratio_xHyL), (pixel_xLyH, ratio_xLyH), (pixel_xHyH, ratio_xHyH)

def makepixel(buffer, x: float, y: float) -> float:
    """ get the interpolated pixel value from buffer at x,y """
    LL, HL, LH, HH = pixelparts(buffer, x, y)
    part_xLyL = LL[0] * LL[1]
    part_xHyL = HL[0] * HL[1]
    part_xLyH = LH[0] * LH[1]
    part_xHyH = HH[0] * HH[1]
    return part_xLyL + part_xHyL + part_xLyH + part_xHyH

def inimage(buffer, x, y, r):
    """ determine if the points at radius R and centred at X, Y are within the image """
    max_x, max_y = size(buffer)
    if (x - r) < 0 or (x + r) >= max_x or (y - r) < 0 or (y + r) >= max_y:
        return False
    else:
        return True

def incolour(buffer, colour=None):
    """ turn buffer into a colour image if required and its not already """
    if colour is None or type(colour) == tuple:
        return colourize(buffer)
    return buffer

def colourize(buffer):
    """ make grey image into an RGB one,
        returns the image array with 3,
        its a no-op if we're not a grey image
        """
    if len(buffer.shape) == 2:
        image = cv2.merge([buffer, buffer, buffer])
    else:
        image = buffer
    return image

def settext(buffer, text, origin, size=0.5, colour=0):
    """ set a text string at x,y of given size and colour (greyscale or a colour tuple) """
    buffer = incolour(buffer, colour)  # colourize iff required
    return cv2.putText(buffer, text, make_int(origin), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)

def circle(buffer, origin, radius, colour=0, thickness=1):
    """ draw a circle as directed """
    buffer = incolour(buffer, colour)  # colourize iff required
    if radius < 1:
        # too small for a circle, do a point instead
        putpixel(buffer, make_int(origin[0]), make_int(origin[1]), colour)
        return buffer
    return cv2.circle(buffer, make_int(origin), make_int(radius), colour, thickness)

def rectangle(buffer, top_left, bottom_right, colour=0, thickness=1):
    """ draw a rectangle as directed """
    buffer = incolour(buffer, colour)  # colourize iff required
    return cv2.rectangle(buffer, make_int(top_left), make_int(bottom_right), colour, thickness)

def line(buffer, from_here, to_there, colour=0, thickness=1):
    """ draw a line as directed """
    buffer = incolour(buffer, colour)  # colourize iff required
    return cv2.line(buffer, make_int(from_here), make_int(to_there), colour, thickness)

def plot(buffer, points, colour):
    """ plot the given points in the given colour """
    buffer = incolour(buffer, colour)  # colourize iff required
    for x, y in points:
        putpixel(buffer, x, y, colour)
    return buffer

def make_int(this):
    """ make the given thing (a number or a tuple of numbers) into integers """
    if type(this) == tuple:
        # Stupid language, cannot create a tuple like this, it makes a 'generator', useless!
        return [int(round(x)) for x in this]
    else:
        return int(round(this))

def translate(point, radius, origin, scale):
    """ translate and scale a circle,
        if origin is +ve map from a full-image to a sub-image (an extraction from the full image),
        if origin is -ve map from a sub-image to a full-image,
        relative to (0,0) to be relative to origin and scale it by scale
        """
    if origin[0] < 0 and origin[1] < 0:
        # map from a sub-image to a full-image
        x = (point[0] / scale) - origin[0]
        y = (point[1] / scale) - origin[1]
        radius /= scale
    elif origin[0] >= 0 and origin[1] >= 0:
        # map from full-image to a sub-image
        x = (point[0] - origin[0]) * scale
        y = (point[1] - origin[1]) * scale
        radius *= scale
    else:
        raise Exception('Origin must be (-ve,-ve) or (+ve,+ve) not a mixture {}'.format(origin))
    return (x, y), radius
