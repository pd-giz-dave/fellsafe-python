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
