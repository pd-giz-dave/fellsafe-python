""" wrapper around cv2 to hide it from everyone
    these functions manipulate a 2D array of luminance values,
    it uses the opencv library to read/write images and numpy arrays to modify images at the pixel level
    """

import numpy as np
import cv2
import const
import utils

""" WARNING
    cv2 and numpy's co-ordinates are backwards from our pov, the 'x' co-ordinate are columns and 'y' rows.
    The functions here use 'x' then 'y' parameters, swapping as required when dealing with cv2 and numpy arrays.
    NB: The final implementation will not use cv2 (its too big) so its use here is purely a diagnostic convenience.
        Significant algorithms are implemented DIY. 
    """

# region Very simple bitmap font...
# Stolen from https://github.com/mikaelpatel/Arduino-LCD/blob/master/src/Driver/PCD8544.h#L304 with minor tweaks
# Only does ASCII chars 32..127 in a 5x7 font in a 6x8 cell

SIMPLE_FONT = [
      0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x5F, 0x00, 0x00,
      0x00, 0x07, 0x00, 0x07, 0x00,
      0x14, 0x7F, 0x14, 0x7F, 0x14,
      0x24, 0x2A, 0x7F, 0x2A, 0x12,
      0x23, 0x13, 0x08, 0x64, 0x62,
      0x36, 0x49, 0x56, 0x20, 0x50,
      0x00, 0x08, 0x07, 0x03, 0x00,
      0x00, 0x1C, 0x22, 0x41, 0x00,
      0x00, 0x41, 0x22, 0x1C, 0x00,
      0x2A, 0x1C, 0x7F, 0x1C, 0x2A,
      0x08, 0x08, 0x3E, 0x08, 0x08,
      0x00, 0x80, 0x70, 0x30, 0x00,
      0x08, 0x08, 0x08, 0x08, 0x08,
      0x00, 0x00, 0x60, 0x60, 0x00,
      0x20, 0x10, 0x08, 0x04, 0x02,
      0x3E, 0x51, 0x49, 0x45, 0x3E,
      0x00, 0x42, 0x7F, 0x40, 0x00,
      0x72, 0x49, 0x49, 0x49, 0x46,
      0x21, 0x41, 0x49, 0x4D, 0x33,
      0x18, 0x14, 0x12, 0x7F, 0x10,
      0x27, 0x45, 0x45, 0x45, 0x39,
      0x3C, 0x4A, 0x49, 0x49, 0x31,
      0x41, 0x21, 0x11, 0x09, 0x07,
      0x36, 0x49, 0x49, 0x49, 0x36,
      0x46, 0x49, 0x49, 0x29, 0x1E,
      0x00, 0x00, 0x14, 0x00, 0x00,
      0x00, 0x40, 0x34, 0x00, 0x00,
      0x00, 0x08, 0x14, 0x22, 0x41,
      0x14, 0x14, 0x14, 0x14, 0x14,
      0x00, 0x41, 0x22, 0x14, 0x08,
      0x02, 0x01, 0x59, 0x09, 0x06,
      0x3E, 0x41, 0x5D, 0x59, 0x4E,
      0x7C, 0x12, 0x11, 0x12, 0x7C,
      0x7F, 0x49, 0x49, 0x49, 0x36,
      0x3E, 0x41, 0x41, 0x41, 0x22,
      0x7F, 0x41, 0x41, 0x41, 0x3E,
      0x7F, 0x49, 0x49, 0x49, 0x41,
      0x7F, 0x09, 0x09, 0x09, 0x01,
      0x3E, 0x41, 0x41, 0x51, 0x73,
      0x7F, 0x08, 0x08, 0x08, 0x7F,
      0x00, 0x41, 0x7F, 0x41, 0x00,
      0x20, 0x40, 0x41, 0x3F, 0x01,
      0x7F, 0x08, 0x14, 0x22, 0x41,
      0x7F, 0x40, 0x40, 0x40, 0x40,
      0x7F, 0x02, 0x1C, 0x02, 0x7F,
      0x7F, 0x04, 0x08, 0x10, 0x7F,
      0x3E, 0x41, 0x41, 0x41, 0x3E,
      0x7F, 0x09, 0x09, 0x09, 0x06,
      0x3E, 0x41, 0x51, 0x21, 0x5E,
      0x7F, 0x09, 0x19, 0x29, 0x46,
      0x26, 0x49, 0x49, 0x49, 0x32,
      0x03, 0x01, 0x7F, 0x01, 0x03,
      0x3F, 0x40, 0x40, 0x40, 0x3F,
      0x1F, 0x20, 0x40, 0x20, 0x1F,
      0x3F, 0x40, 0x38, 0x40, 0x3F,
      0x63, 0x14, 0x08, 0x14, 0x63,
      0x03, 0x04, 0x78, 0x04, 0x03,
      0x61, 0x59, 0x49, 0x4D, 0x43,
      0x00, 0x7F, 0x41, 0x41, 0x41,
      0x02, 0x04, 0x08, 0x10, 0x20,
      0x00, 0x41, 0x41, 0x41, 0x7F,
      0x04, 0x02, 0x01, 0x02, 0x04,
      0x40, 0x40, 0x40, 0x40, 0x40,
      0x00, 0x03, 0x07, 0x08, 0x00,
      0x20, 0x54, 0x54, 0x78, 0x40,
      0x7F, 0x28, 0x44, 0x44, 0x38,
      0x38, 0x44, 0x44, 0x44, 0x28,
      0x38, 0x44, 0x44, 0x28, 0x7F,
      0x38, 0x54, 0x54, 0x54, 0x18,
      0x00, 0x08, 0x7E, 0x09, 0x02,
      0x18, 0xA4, 0xA4, 0x9C, 0x78,
      0x7F, 0x08, 0x04, 0x04, 0x78,
      0x00, 0x44, 0x7D, 0x40, 0x00,
      0x20, 0x40, 0x40, 0x3D, 0x00,
      0x7F, 0x10, 0x28, 0x44, 0x00,
      0x00, 0x41, 0x7F, 0x40, 0x00,
      0x7C, 0x04, 0x78, 0x04, 0x78,
      0x7C, 0x08, 0x04, 0x04, 0x78,
      0x38, 0x44, 0x44, 0x44, 0x38,
      0xFC, 0x18, 0x24, 0x24, 0x18,
      0x18, 0x24, 0x24, 0x18, 0xFC,
      0x7C, 0x08, 0x04, 0x04, 0x08,
      0x48, 0x54, 0x54, 0x54, 0x24,
      0x04, 0x04, 0x3F, 0x44, 0x24,
      0x3C, 0x40, 0x40, 0x20, 0x7C,
      0x1C, 0x20, 0x40, 0x20, 0x1C,
      0x3C, 0x40, 0x30, 0x40, 0x3C,
      0x44, 0x28, 0x10, 0x28, 0x44,
      0x0C, 0x90, 0x90, 0x90, 0x7C,
      0x44, 0x64, 0x54, 0x4C, 0x44,
      0x00, 0x08, 0x36, 0x41, 0x00,
      0x00, 0x00, 0x7F, 0x00, 0x00,
      0x00, 0x41, 0x36, 0x08, 0x00,
      0x02, 0x01, 0x02, 0x04, 0x02,
      0x3C, 0x26, 0x23, 0x26, 0x3C,
    ]
SIMPLE_FONT_CHAR_HEIGHT = 7
SIMPLE_FONT_CHAR_WIDTH  = 5
# endregion

# region cv2 usage...
def load(image_file):
    """ load a buffer from an image file as a grey scale image and return it """
    return cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

def unload(buffer, image_file):
    """ unload the given buffer to a PNG image file """
    cv2.imwrite(image_file, buffer)

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

# endregion

def copy(buffer):
    """ return a copy of the given buffer """
    return np.copy(buffer)

def new(width, height, luminance=const.MIN_LUMINANCE):
    """ prepare a new buffer of the given size and luminance """
    return np.full((height, width), luminance, dtype=np.uint8)  # NB: numpy arrays follow cv2 conventions

def size(buffer):
    """ return the x,y size of the given buffer """
    max_x = buffer.shape[1]  # NB: cv2/numpy x,y are reversed
    max_y = buffer.shape[0]  # ..
    return max_x, max_y

def downsize(buffer, new_size):
    """ downsize the given greyscale image such that its width is at most that given,
        the aspect ratio is preserved, its a no-op if image already small enough,
        returns a new buffer of the new size,
        this is purely a diagnostic aid to simulate low-resolution cameras,
        as such it uses a very simple algorithm:
          it calculates the sub-image size in the original image that must be represented in the downsized image
          the downsized image pixel is then just the average of the original sub-image pixels
        the assumption is that the light that fell on the pixels in the original sub-image would all of have
        fallen on a single pixel in a 'lesser' camera, so the lesser camera would only have seen their average,
        the averages are calculated from the integral of the original image, this means each pixel of the original
        image is only visited once, and the average over any arbitrary area only requires four accesses of that
        for each downsized pixel.
        """
    width, height = size(buffer)
    if width <= new_size:
        # its already small enough
        return buffer
    # bring width down to new size and re-scale height to maintain aspect ratio
    new_width    = new_size
    width_scale  = width / new_width
    new_height   = int(height / width_scale)
    height_scale = height / new_height
    # calculate the kernel size for our average
    kernel_width   = int(round(width_scale))    # guaranteed to be >= 1
    kernel_height  = int(round(height_scale))   # ..
    kernel_plus_x  = kernel_width >> 1    # offset for going forward
    kernel_minus_x = kernel_plus_x - 1    # offset for going backward
    kernel_plus_y  = kernel_height >> 1   # ditto
    kernel_minus_y = kernel_plus_y - 1    # ..
    # do the downsize via the integral
    integral  = integrate(buffer)
    downsized = np.zeros((new_height, new_width), np.uint8)
    for x in range(new_width):
        orig_x  = int(min(x * width_scale, width - 1))
        orig_x1 = int(max(orig_x - kernel_minus_x, 0))
        orig_x2 = int(min(orig_x + kernel_plus_x, width - 1))
        for y in range(new_height):
            orig_y  = int(min(y * height_scale, height - 1))
            orig_y1 = int(max(orig_y - kernel_minus_y, 0))
            orig_y2 = int(min(orig_y + kernel_plus_y, height - 1))
            count = int((orig_x2 - orig_x1) * (orig_y2 - orig_y1))  # how many samples in the integration area
            # sum = bottom right (x2,y2) + top left (x1,y1) - top right (x2,y1) - bottom left (x1,y2)
            # where all but bottom right are *outside* the integration window
            average = int((integral[orig_y2][orig_x2] + integral[orig_y1][orig_x1]
                          -
                          (integral[orig_y1][orig_x2] + integral[orig_y2][orig_x1])) / count)
            downsized[y][x] = average
    return downsized

def prepare(src, width, logger=None):
    """ load and downsize an image """
    if logger is not None:
        logger.log('Preparing image to width {} from {}'.format(width, src))
    source = load(src)
    if source is None:
        if logger is not None:
            logger.log('Cannot load {}'.format(src))
        return None
    # Downsize it (to simulate low quality smartphone cameras)
    downsized = downsize(source, width)
    if logger is not None:
        logger.log('Original size {} reduced to {}'.format(size(source), size(downsized)))
        logger.draw(downsized, file='downsized')
    return downsized

def integrate(buffer, box=None):
    """ generate the integral of the given box within the given image buffer """

    if box is None:
        box_min_x, box_min_y = (0, 0)
        box_max_x, box_max_y = size(buffer)
    else:
        box_min_x = box[0][0]
        box_max_x = box[1][0] + 1
        box_min_y = box[0][1]
        box_max_y = box[1][1] + 1
    box_width  = box_max_x - box_min_x
    box_height = box_max_y - box_min_y

    # make an empty buffer to accumulate our integral in
    integral = np.zeros((box_height, box_width), np.uint32)

    for y in range(box_min_y, box_max_y):
        for x in range(box_min_x, box_max_x):
            if x == box_min_x:
                acc = int(buffer[y, x])  # start a new row (# NB: need int 'cos source is uint8, i.e. unsigned)
            else:
                acc += int(buffer[y, x])  # extend existing row (# NB: need int 'cos source is uint8, i.e. unsigned)
            ix = x - box_min_x
            iy = y - box_min_y
            if iy == 0:
                integral[iy][ix] = acc  # start a new column
            else:
                integral[iy][ix] = acc + integral[iy - 1][ix]  # extend existing column

    return integral

def blur(buffer, kernel_size):
    """ return the blurred image as a mean blur over the given kernel size """
    # we do this by integrating then calculating the average via integral differences,
    # this means we only visit each pixel once irrespective of the kernel size
    if kernel_size is None or kernel_size < 2:
        return buffer  # pointless

    # integrate the image
    integral = integrate(buffer)

    # get image geometry
    width  = len(integral[0])
    height = len(integral)

    # set kernel geometry
    kernel_plus  = kernel_size >> 1  # offset for going forward
    kernel_minus = kernel_size - 1   # offset for going backward

    # blur the image by averaging over the given kernel size
    blurred = np.zeros((height, width), np.uint8)
    for x in range(width):
        x1 = int(max(x - kernel_minus, 0))
        x2 = int(min(x + kernel_plus, width - 1))
        for y in range(height):
            y1 = int(max(y - kernel_minus, 0))
            y2 = int(min(y + kernel_plus, height - 1))
            count = int((x2 - x1) * (y2 - y1))  # how many samples in the integration area
            # sum = bottom right (x2,y2) + top left (x1,y1) - top right (x2,y1) - bottom left (x1,y2)
            # where all but bottom right are *outside* the integration window
            average = int((integral[y2][x2] + integral[y1][x1] - integral[y1][x2] - integral[y2][x1]) / count)
            blurred[y][x] = average

    return blurred

def get_box_corners(buffer, box=None):
    """ get the co-ordinates of a box in the buffer """
    buffer_min_x, buffer_min_y = (0, 0)
    buffer_max_x, buffer_max_y = size(buffer)
    if box is None:
        box_min_x = buffer_min_x
        box_max_x = buffer_max_x
        box_min_y = buffer_min_y
        box_max_y = buffer_max_y
    else:
        box_min_x = box[0][0]
        box_max_x = box[1][0] + 1
        box_min_y = box[0][1]
        box_max_y = box[1][1] + 1
    return (box_min_x, box_min_y), (box_max_x, box_max_y)

def binarize(buffer, box=None, width: float=8, height: float=None, black: float=15, white: float=None):
    """ create a binary (or tertiary) image of the source image within the box using an adaptive threshold,
        if box is None the whole image is processed, otherwise just the area within the given box,
        width is the fraction of the source/box width to use as the integration area,
        height is the fraction of the source/box height to use as the integration area (None==same as width in pixels)
        black is the % below the average that is considered to be the black/grey boundary,
        white is the % above the average that is considered to be the grey/white boundary,
        white of None means same as black and will yield a binary image,
        See the adaptive-threshold-algorithm.pdf paper for algorithm details.
        the image returned is the same size as the box (or the source iff no box given)
        """

    # region get the source and box metrics...
    (box_min_x, box_min_y), (box_max_x, box_max_y) = get_box_corners(buffer, box)
    box_width  = box_max_x - box_min_x
    box_height = box_max_y - box_min_y
    # endregion

    # region set the integration size...
    width_pixels = int(box_width / width)            # we want this to be odd so that there is a centre
    width_plus   = max(width_pixels >> 1, 2)         # offset for going forward
    width_minus  = width_plus - 1                    # offset for going backward
    if height is None:
        height_pixels = width_pixels                 # make it square
    else:
        height_pixels = int(box_height / height)     # we want this to be odd so that there is a centre
    height_plus  = max(height_pixels >> 1, 2)        # offset for going forward
    height_minus = height_plus - 1                   # offset for going backward
    # endregion

    # integrate the image
    integral = integrate(buffer, box)

    # region do the threshold on a new image buffer...
    binary = np.zeros((box_height, box_width), np.uint8)
    black_limit = (100-black)/100    # convert % to a ratio
    if white is None:
        white_limit = black_limit
    else:
        white_limit = (100+white)/100  # convert % to a ratio
    for x in range(box_width):
        x1 = int(max(x - width_minus, 0))
        x2 = int(min(x + width_plus, box_width - 1))
        for y in range(box_height):
            y1 = int(max(y - height_minus, 0))
            y2 = int(min(y + height_plus, box_height - 1))
            count = int((x2 - x1) * (y2 - y1))  # how many samples in the integration area
            # sum = bottom right (x2,y2) + top left (x1,y1) - top right (x2,y1) - bottom left (x1,y2)
            # where all but bottom right are *outside* the integration window
            acc = integral[y2][x2] + integral[y1][x1] - integral[y1][x2] - integral[y2][x1]
            src = int(buffer[box_min_y + y, box_min_x + x]) * count  # NB: need int 'cos source is uint8 (unsigned)
            if src >= (acc * white_limit):
                binary[y, x] = const.MAX_LUMINANCE
            elif src <= (acc * black_limit):
                binary[y, x] = const.MIN_LUMINANCE
            else:
                binary[y, x] = const.MID_LUMINANCE
    # endregion

    return binary

def extract(image, box: ((int, int), (int, int))):
    """ extract a box from the given image """
    tl_x, tl_y = box[0]
    br_x, br_y = box[1]
    width  = br_x - tl_x + 1
    height = br_y - tl_y + 1
    buffer = np.zeros((height, width), np.uint8)
    for x in range(width):
        for y in range(height):
            buffer[y, x] = image[tl_y + y, tl_x + x]
    return buffer

def histogram(buffer, box=None):
    """ build a histogram of luminance values in the given buffer (or the box within it) """
    (tl_x, tl_y), (br_x, br_y) = get_box_corners(buffer, box)
    width  = br_x - tl_x
    height = br_y - tl_y
    samples = [0 for _ in range(const.MAX_LUMINANCE+1)]
    for x in range(width):
        for y in range(height):
            samples[buffer[tl_y + y, tl_x + x]] += 1
    return samples

def upsize(buffer, scale: float):
    """ return a buffer that is scale times bigger in width and height than that given,
        the contents of the given buffer are also scaled by interpolating neighbours (using makepixel),
        scale must be positive and greater than 1,
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

def settext(buffer, text, origin, colour=0):
    """ set a text string at x,y of given colour (greyscale or a colour tuple),
        a very simple 5x7 bitmap font is used (good enough for our purposes)
        """
    buffer = incolour(buffer, colour)      # colourize iff required
    cursor_x, cursor_y = make_int(origin)  # this is the bottom left of the text, i.e. the 'baseline'
    cursor_y -= SIMPLE_FONT_CHAR_HEIGHT    # move it to the top
    start_x = cursor_x                     # used for new-lines (which we allow)
    for char in text:
        if char == '\n':
            cursor_x  = start_x
            cursor_y += SIMPLE_FONT_CHAR_HEIGHT + 1  # +1 for the inter-line gap
            continue
        char_index = (ord(char) - ord(' ')) * SIMPLE_FONT_CHAR_WIDTH
        if char_index < 0 or char_index >= len(SIMPLE_FONT):
            # ignore control chars and out-of-range chars
            continue
        cols = SIMPLE_FONT[char_index : char_index + SIMPLE_FONT_CHAR_WIDTH]
        for col, bits in enumerate(cols):
            for row in range(SIMPLE_FONT_CHAR_HEIGHT):
                if ((bits >> row) & 1) == 1:
                    # draw this bit
                    x = cursor_x + col
                    y = cursor_y + row
                    putpixel(buffer, x, y, colour)
        cursor_x += SIMPLE_FONT_CHAR_WIDTH + 1  # +1 for the inter-char gap
    return buffer

def line(buffer, from_here, to_there, colour=0):
    """ draw a line as directed """
    buffer = incolour(buffer, colour)  # colourize iff required
    points = utils.line(int(round(from_here[0])), int(round(from_here[1])),
                        int(round(to_there[0])), int(round(to_there[1])))
    for point in points:
        putpixel(buffer, point[0], point[1], colour)
    return buffer

def circle(buffer, origin, radius, colour=0):
    """ draw a circle as directed """
    buffer = incolour(buffer, colour)  # colourize iff required
    if radius < 1:
        # too small for a circle, do a point instead
        putpixel(buffer, make_int(origin[0]), make_int(origin[1]), colour)
        return buffer
    points = utils.circumference(origin[0], origin[1], radius)
    for point in points:
        putpixel(buffer, point[0], point[1], colour)
    return buffer

def rectangle(buffer, top_left, bottom_right, colour=0):
    """ draw a rectangle as directed (as four lines) """
    buffer = incolour(buffer, colour)  # colourize iff required
    top_right   = bottom_right[0], top_left[1]
    bottom_left = top_left[0], bottom_right[1]
    buffer = line(buffer, top_left, top_right, colour)
    buffer = line(buffer, top_right, bottom_right, colour)
    buffer = line(buffer, bottom_right, bottom_left, colour)
    buffer = line(buffer, bottom_left, top_left, colour)
    return buffer

def grid(buffer, x_spacing=None, y_spacing=None, colour=0, thickness=1, labels=None, label_spacing=1):
    """ draw grid lines as directed """
    max_x, max_y = size(buffer)
    if x_spacing is not None:
        for x in range(0, max_x, x_spacing):
            buffer = line(buffer, (x, 0), (x, max_y-1), colour=colour)
            if labels is not None:
                if (x % label_spacing) == 0:
                    buffer = settext(buffer, '{}'.format(x), (x, max_y-1), colour=labels)
    if y_spacing is not None:
        for y in range(0, max_y, y_spacing):
            buffer = line(buffer, (0, y), (max_x-1, y), colour=colour)
            if labels is not None:
                if (y % label_spacing) == 0:
                    buffer = settext(buffer, '{}'.format(y), (0, y), colour=labels)
    return buffer

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
